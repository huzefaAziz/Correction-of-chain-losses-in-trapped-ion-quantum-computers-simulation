"""
Simple demonstration of chain loss correction

A minimal example showing the key concepts without running a full simulation.
"""

from chain_loss_simulation import ChainLossSimulator, QubitState
from bb_code import BBCode
import numpy as np


def simple_demo():
    """Simple demonstration of chain loss correction"""
    
    print("=" * 60)
    print("Simple Chain Loss Correction Demo")
    print("=" * 60)
    print()
    
    # Initialize a small BB code
    print("1. Initializing BB code...")
    code = BBCode(n=72, k=12, d=6, num_chains=12, qubits_per_chain=6)
    print(f"   [OK] Code initialized: [[{code.n},{code.k},{code.d}]]")
    print(f"   [OK] Distributed over {code.num_chains} chains")
    print(f"   [OK] {code.qubits_per_chain} qubits per chain")
    print()
    
    # Initialize simulator
    print("2. Initializing simulator...")
    simulator = ChainLossSimulator(
        num_chains=12,
        qubits_per_chain=6,
        data_qubits_per_chain=4,
        ancilla_qubits_per_chain=1,
        beacon_qubits_per_chain=1,
        two_qubit_gate_error_rate=2e-3,
        alpha=1.9,
        rng_seed=42
    )
    print(f"   [OK] Simulator initialized")
    print(f"   [OK] Chain loss rate: {simulator.chain_loss_rate:.2e}")
    print(f"   [OK] Gate error rate: {simulator.two_qubit_gate_error_rate:.2e}")
    print()
    
    # Demonstrate beacon measurement
    print("3. Demonstrating beacon qubit measurement...")
    print("   Measuring beacon qubits in all chains:")
    detections = simulator.measure_beacons_all_chains(time_step=0)
    for chain_id, detected in detections.items():
        status = "LOST" if detected else "OK"
        beacon_qubit = simulator.chains[chain_id].beacon_qubits[0]
        state = simulator.qubit_states[beacon_qubit]
        print(f"   Chain {chain_id}: Beacon = {state.name}, Status = {status}")
    print()
    
    # Simulate a chain loss
    print("4. Simulating chain loss...")
    print("   Attempting to lose chain 5 at time step 10...")
    loss_occurred = simulator.simulate_chain_loss(chain_id=5, time_step=10)
    if loss_occurred:
        print(f"   [OK] Chain 5 lost at time step 10")
        chain = simulator.chains[5]
        print(f"   [OK] Chain status: is_lost={chain.is_lost}")
        print(f"   [OK] Qubits in chain 5: {len(chain.get_all_qubits())}")
        
        # Check qubit states
        erased_count = sum(1 for q in chain.get_all_qubits() 
                          if simulator.qubit_states.get(q) == QubitState.ERASED)
        print(f"   [OK] Erased qubits: {erased_count}/{len(chain.get_all_qubits())}")
    else:
        print("   (No loss occurred - try with higher loss rate)")
    print()
    
    # Demonstrate beacon detection after loss
    print("5. Detecting chain loss with beacon...")
    if simulator.chains[5].is_lost:
        outcome, detected = simulator.measure_beacon(chain_id=5, time_step=11)
        print(f"   Beacon measurement outcome: {'1' if outcome else '0'}")
        print(f"   Chain loss detected: {detected}")
    print()
    
    # Show statistics
    print("6. Simulation statistics:")
    stats = simulator.get_statistics()
    print(f"   Total chain losses: {stats['total_chain_losses']}")
    print(f"   Total operations: {stats['total_operations']}")
    print(f"   Erased qubits: {stats['erased_qubits']}")
    print()
    
    # Show code distribution
    print("7. Code distribution (first 4 chains):")
    for chain_id in range(min(4, code.num_chains)):
        qubits = code.get_qubits_in_chain(chain_id)
        print(f"   Chain {chain_id}: {len(qubits)} qubits (indices {qubits[:3]}...)")
    print()
    
    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == '__main__':
    simple_demo()

