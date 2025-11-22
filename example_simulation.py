"""
Example Simulation: Chain Loss Correction for Trapped Ions

This script demonstrates how to use the chain loss correction simulation
from the paper "Correction of chain losses in trapped ion quantum computers".

It simulates a [[72,12,6]] BB code distributed over 12 chains with beacon qubits.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from typing import List, Tuple
import time
import sys

from chain_loss_simulation import ChainLossSimulator
from erasure_decoder import ErasureDecoder
from bb_code import BBCode


def run_syndrome_extraction_round(
    simulator: ChainLossSimulator,
    code: BBCode,
    decoder: ErasureDecoder,
    time_step: int,
    measurement_order: str = 'X_then_Z'
) -> Tuple[List[int], bool]:
    """
    Run one round of syndrome extraction.
    
    Args:
        simulator: Chain loss simulator
        code: BB code structure
        decoder: Erasure decoder
        time_step: Current time step
        measurement_order: 'X_then_Z' or 'Z_then_X'
    
    Returns:
        (erased_qubits, logical_error)
    """
    # 1. Measure beacon qubits to detect chain losses
    beacon_detections = simulator.measure_beacons_all_chains(time_step)
    
    # 2. Replace lost chains
    erased_qubits = set()
    lost_chain_ids = []
    for chain_id, detected in beacon_detections.items():
        if detected:
            lost_chain_ids.append(chain_id)
            # Replace lost chain (convert to erasure)
            simulator.replace_lost_chain(chain_id, time_step)
            # Get erased qubits from this chain
            chain_qubits = code.get_qubits_in_chain(chain_id)
            erased_qubits.update(chain_qubits)
    
    # 3. Simulate syndrome extraction circuit
    # This is simplified - in practice would have full stabilizer measurement circuit
    syndromes = []
    
    # Measure X stabilizers
    if measurement_order == 'X_then_Z':
        for stab_id in range(len(code.stabilizers) // 2):
            stabilizer_qubits = code.get_stabilizer_support(stab_id)
            
            # Check if any qubit is erased
            is_erased = bool(set(stabilizer_qubits) & erased_qubits)
            
            # Simplified syndrome measurement
            # In practice, would execute full measurement circuit
            outcome = simulator.rng.choice([True, False])  # +1 or -1 eigenvalue
            if is_erased:
                outcome = None  # Erased measurement
            
            decoder.add_syndrome(stab_id, outcome, time_step, is_erased)
            
            if outcome is not None:
                syndromes.append((stab_id, outcome, is_erased))
        
        # Measure Z stabilizers (alternate with X)
        for stab_id in range(len(code.stabilizers) // 2, len(code.stabilizers)):
            stabilizer_qubits = code.get_stabilizer_support(stab_id)
            is_erased = bool(set(stabilizer_qubits) & erased_qubits)
            outcome = simulator.rng.choice([True, False])
            if is_erased:
                outcome = None
            
            decoder.add_syndrome(stab_id, outcome, time_step, is_erased)
            
            if outcome is not None:
                syndromes.append((stab_id, outcome, is_erased))
    
    # 4. Simulate chain losses during circuit execution
    for chain_id in range(simulator.num_chains):
        simulator.simulate_chain_loss(chain_id, time_step)
    
    return (list(erased_qubits), False)


def simulate_chain_loss_correction(
    num_rounds: int = 6,
    num_timesteps_per_round: int = 80,
    two_qubit_gate_error_rate: float = 1e-3,
    alpha: float = 1.9,
    num_chains: int = 12,
    code_params: dict = None
) -> dict:
    """
    Run full simulation of chain loss correction.
    
    Args:
        num_rounds: Number of syndrome extraction rounds
        num_timesteps_per_round: Time steps per round
        two_qubit_gate_error_rate: Physical gate error rate p
        alpha: Exponent for chain loss rate (p_loss = p^alpha)
        num_chains: Number of ion chains
        code_params: BB code parameters
    
    Returns:
        Dictionary with simulation results
    """
    if code_params is None:
        code_params = {'n': 72, 'k': 12, 'd': 6}
    
    # Initialize BB code
    code = BBCode(
        n=code_params['n'],
        k=code_params['k'],
        d=code_params['d'],
        num_chains=num_chains,
        qubits_per_chain=code_params['n'] // num_chains
    )
    
    # Initialize decoder
    decoder = ErasureDecoder(
        code_stabilizers=code.stabilizers,
        code_checks=[],  # Simplified - would have full check matrix
        logical_operators=code.logical_operators
    )
    
    # Initialize simulator
    chain_loss_rate = two_qubit_gate_error_rate ** alpha
    simulator = ChainLossSimulator(
        num_chains=num_chains,
        qubits_per_chain=code_params['n'] // num_chains,
        data_qubits_per_chain=code_params['n'] // num_chains - 2,  # Reserve for ancilla/beacon
        ancilla_qubits_per_chain=1,
        beacon_qubits_per_chain=1,
        two_qubit_gate_error_rate=two_qubit_gate_error_rate,
        measurement_error_rate=two_qubit_gate_error_rate,
        chain_loss_rate=chain_loss_rate,
        alpha=alpha,
        rng_seed=42
    )
    
    # Run simulation
    results = {
        'logical_errors': [],
        'chain_losses': [],
        'erased_qubits_history': [],
        'rounds': []
    }
    
    total_timestep = 0
    for round_num in range(num_rounds):
        print(f"Running round {round_num + 1}/{num_rounds}...")
        
        for timestep_in_round in range(num_timesteps_per_round):
            time_step = round_num * num_timesteps_per_round + timestep_in_round
            
            # Run syndrome extraction
            erased_qubits, logical_error = run_syndrome_extraction_round(
                simulator,
                code,
                decoder,
                time_step,
                measurement_order='X_then_Z'
            )
            
            results['erased_qubits_history'].append(len(erased_qubits))
            
            total_timestep += 1
        
        # Decode at end of round
        erased_qubits = simulator.get_erased_qubits()
        syndromes = decoder.syndromes[-num_timesteps_per_round:]
        
        correction_qubits, logical_error = decoder.decode_with_erasures(
            syndromes,
            erased_qubits
        )
        
        results['logical_errors'].append(logical_error)
        results['rounds'].append(round_num + 1)
        
        # Reset decoder for next round (or keep history)
        # decoder.reset()
    
    # Get final statistics
    stats = simulator.get_statistics()
    results.update(stats)
    
    # Make sure chain_losses are properly included
    if 'chain_losses' in stats:
        results['chain_losses'] = stats['chain_losses']
    elif 'chain_losses' not in results:
        results['chain_losses'] = []
    
    # Add num_chains for plotting
    results['num_chains'] = num_chains
    
    # Debug: Print chain losses if any occurred
    if results['chain_losses']:
        print(f"Chain losses occurred: {len(results['chain_losses'])} events")
        for chain_id, time_step in results['chain_losses'][:5]:  # Show first 5
            print(f"  Chain {chain_id} lost at time step {time_step}")
    else:
        expected = chain_loss_rate * num_rounds * num_timesteps_per_round * num_chains
        print(f"No chain losses occurred (expected ~{expected:.3f} events)")
    
    return results


def plot_simulation_results(results: dict, save_path: str = None):
    """
    Plot simulation results.
    
    Args:
        results: Simulation results dictionary
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Logical errors per round
    ax1 = axes[0, 0]
    ax1.plot(results['rounds'], results['logical_errors'], 'o-', label='Logical Error')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Logical Error (1=error, 0=no error)')
    ax1.set_title('Logical Errors per Round')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Erased qubits over time
    ax2 = axes[0, 1]
    ax2.plot(results['erased_qubits_history'], alpha=0.7, label='Erased Qubits')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Number of Erased Qubits')
    ax2.set_title('Erased Qubits Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Chain losses or simulation timeline
    ax3 = axes[1, 0]
    chain_losses = results.get('chain_losses', [])
    num_chains = results.get('num_chains', 12)
    total_timesteps = len(results.get('erased_qubits_history', []))
    
    if chain_losses and len(chain_losses) > 0:
        # Plot actual chain losses
        chain_loss_times = [t for _, t in chain_losses]
        chain_loss_ids = [c for c, _ in chain_losses]
        ax3.scatter(chain_loss_times, chain_loss_ids, alpha=0.7, s=60, color='red', 
                   marker='x', linewidths=2, label=f'Chain Loss (n={len(chain_losses)})')
        ax3.set_xlabel('Time Step', fontsize=10)
        ax3.set_ylabel('Chain ID', fontsize=10)
        ax3.set_title(f'Chain Loss Events (Total: {len(chain_losses)})', fontsize=11, fontweight='bold')
        ax3.set_xlim(-1, max(total_timesteps, max(chain_loss_times) + 1) if chain_loss_times else total_timesteps)
        ax3.set_ylim(-0.5, num_chains - 0.5)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        # Show simulation timeline with expected loss rate
        chain_loss_rate = results.get('chain_loss_rate', 0)
        total_timesteps = len(results.get('erased_qubits_history', []))
        expected_losses = chain_loss_rate * total_timesteps * num_chains
        
        # Plot timeline showing when losses could occur
        ax3.plot([0, total_timesteps], [num_chains/2, num_chains/2], 
                '--', alpha=0.3, color='gray', linewidth=1, label='Simulation Timeline')
        
        # Show statistics in plot
        info_text = f"""No chain losses occurred
        
Chain Loss Rate: {chain_loss_rate:.2e}
Expected Losses: {expected_losses:.3f}
Time Steps: {total_timesteps}
Chains: {num_chains}

(Chain loss rate is very low,
so losses are rare in this run)"""
        
        ax3.text(0.5, 0.45, info_text, 
                ha='center', va='center', transform=ax3.transAxes, 
                fontsize=9, family='monospace', color='black',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        ax3.set_xlabel('Time Step', fontsize=10)
        ax3.set_ylabel('Chain ID', fontsize=10)
        ax3.set_title('Chain Loss Events', fontsize=11, fontweight='bold')
        ax3.set_xlim(-1, max(total_timesteps, 1))
        ax3.set_ylim(0, num_chains)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistics summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
    Simulation Statistics
    
    Total Chain Losses: {results.get('total_chain_losses', 0)}
    Total Operations: {results.get('total_operations', 0)}
    Chain Loss Rate: {results.get('chain_loss_rate', 0):.2e}
    Gate Error Rate: {results.get('gate_error_rate', 0):.2e}
    
    Logical Error Rate: {np.mean(results['logical_errors']):.3f}
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def main():
    """Main function to run example simulation"""
    print("=" * 60)
    print("Chain Loss Correction Simulation")
    print("Based on arXiv:2511.16632")
    print("=" * 60)
    print()
    
    # Simulation parameters
    params = {
        'num_rounds': 6,
        'num_timesteps_per_round': 80,
        'two_qubit_gate_error_rate': 2e-3,
        'alpha': 1.9,
        'num_chains': 12,
        'code_params': {'n': 72, 'k': 12, 'd': 6}
    }
    
    print("Simulation parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print()
    
    # Run simulation
    print("Running simulation...")
    sys.stdout.flush()
    start_time = time.time()
    
    results = simulate_chain_loss_correction(**params)
    
    elapsed_time = time.time() - start_time
    print(f"Simulation completed in {elapsed_time:.2f} seconds")
    sys.stdout.flush()
    print()
    
    # Print results
    print("Results:")
    print(f"  Total chain losses: {results['total_chain_losses']}")
    print(f"  Logical error rate: {np.mean(results['logical_errors']):.3f}")
    print(f"  Average erased qubits: {np.mean(results['erased_qubits_history']):.2f}")
    sys.stdout.flush()
    print()
    
    # Plot results
    print("Generating plots...")
    sys.stdout.flush()
    plot_simulation_results(results, save_path='chain_loss_simulation_results.png')
    print("Done!")
    sys.stdout.flush()


if __name__ == '__main__':
    main()

