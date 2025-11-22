"""
Visualization utilities for chain loss correction simulations

Provides functions to visualize:
- Chain loss events over time
- Erased qubits distribution
- Logical error rates
- Beacon detection efficiency
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def plot_chain_loss_events(
    chain_losses: List[Tuple[int, int]],
    num_chains: int,
    total_timesteps: int,
    title: str = "Chain Loss Events",
    save_path: Optional[str] = None
):
    """
    Plot chain loss events over time.
    
    Args:
        chain_losses: List of (chain_id, time_step) tuples
        num_chains: Total number of chains
        total_timesteps: Total number of time steps
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if chain_losses:
        times = [t for _, t in chain_losses]
        chain_ids = [c for c, _ in chain_losses]
        
        ax.scatter(times, chain_ids, alpha=0.6, s=50, color='red', label='Chain Loss')
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Chain ID', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(-0.5, num_chains - 0.5)
        ax.set_xlim(-1, total_timesteps + 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No chain losses occurred', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def plot_logical_error_rate_vs_gate_error(
    gate_error_rates: List[float],
    logical_error_rates: List[float],
    alpha_values: List[float] = None,
    title: str = "Logical Error Rate vs Gate Error Rate",
    save_path: Optional[str] = None
):
    """
    Plot logical error rate as a function of gate error rate.
    
    Similar to Figure 3 in the paper.
    
    Args:
        gate_error_rates: List of gate error rates p
        logical_error_rates: List of corresponding logical error rates
        alpha_values: List of alpha values for p_loss = p^alpha (optional)
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(gate_error_rates, logical_error_rates, 'o-', linewidth=2, markersize=8)
    
    # Add threshold line (break-even point)
    threshold = min(gate_error_rates)
    ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, 
               label=f'Break-even threshold (p = {threshold:.2e})')
    
    ax.set_xlabel('Physical 2-Qubit Gate Error Rate $p$', fontsize=12)
    ax.set_ylabel('Logical Error Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def plot_erased_qubits_distribution(
    erased_qubits_history: List[int],
    num_chains: int,
    title: str = "Erased Qubits Over Time",
    save_path: Optional[str] = None
):
    """
    Plot distribution of erased qubits over time.
    
    Args:
        erased_qubits_history: List of number of erased qubits at each time step
        num_chains: Total number of chains
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Time series plot
    ax1 = axes[0]
    ax1.plot(erased_qubits_history, alpha=0.7, linewidth=1)
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Number of Erased Qubits', fontsize=12)
    ax1.set_title(f'{title} - Time Series', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Histogram
    ax2 = axes[1]
    ax2.hist(erased_qubits_history, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Erased Qubits', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Erased Qubits', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def plot_beacon_detection_efficiency(
    beacon_detections: List[Tuple[int, int, bool]],
    chain_losses: List[Tuple[int, int]],
    title: str = "Beacon Detection Efficiency",
    save_path: Optional[str] = None
):
    """
    Plot beacon detection efficiency.
    
    Compares detected chain losses vs actual chain losses.
    
    Args:
        beacon_detections: List of (chain_id, time, detected) tuples
        chain_losses: List of (chain_id, time) tuples for actual losses
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if not chain_losses:
        ax.text(0.5, 0.5, 'No chain losses to analyze', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        return
    
    # Create detection timeline
    loss_times = sorted(set([t for _, t in chain_losses]))
    detected_losses = [d for _, _, d in beacon_detections if d]
    
    # Calculate detection efficiency
    total_losses = len(chain_losses)
    total_detected = len(detected_losses)
    efficiency = total_detected / total_losses if total_losses > 0 else 0.0
    
    # Plot
    ax.bar(['Detected', 'Missed'], 
           [total_detected, total_losses - total_detected],
           color=['green', 'red'], alpha=0.7)
    
    ax.set_ylabel('Number of Chain Losses', fontsize=12)
    ax.set_title(f'{title}\nEfficiency: {efficiency:.1%}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add text annotations
    for i, (label, value) in enumerate(zip(['Detected', 'Missed'], 
                                           [total_detected, total_losses - total_detected])):
        ax.text(i, value + 0.5, str(value), ha='center', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def plot_code_distribution(
    code_distribution: Dict[int, List[int]],
    logical_operators: Dict[str, List[List[int]]],
    title: str = "Code Distribution Over Chains",
    save_path: Optional[str] = None
):
    """
    Visualize how the code is distributed over chains.
    
    Args:
        code_distribution: Dictionary mapping chain_id to qubit indices
        logical_operators: Dictionary with 'X' and 'Z' logical operators
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    num_chains = len(code_distribution)
    max_qubits = max(len(qubits) for qubits in code_distribution.values())
    
    # Plot qubit distribution
    ax1 = axes[0]
    chain_ids = list(code_distribution.keys())
    qubit_counts = [len(code_distribution[cid]) for cid in chain_ids]
    
    ax1.bar(chain_ids, qubit_counts, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Chain ID', fontsize=12)
    ax1.set_ylabel('Number of Qubits', fontsize=12)
    ax1.set_title('Qubits per Chain', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot logical operator distribution (simplified)
    ax2 = axes[1]
    if logical_operators:
        # Count how many qubits of each logical operator are in each chain
        logical_dist = {}
        for op_type in ['X', 'Z']:
            for logical_idx, logical_op in enumerate(logical_operators.get(op_type, [])):
                logical_qubits = set(logical_op)
                for chain_id, chain_qubits in code_distribution.items():
                    key = f"{op_type}_{logical_idx}"
                    if key not in logical_dist:
                        logical_dist[key] = {}
                    overlap = len(logical_qubits & set(chain_qubits))
                    logical_dist[key][chain_id] = overlap
        
        # Visualize one logical operator as example
        if logical_dist:
            example_key = list(logical_dist.keys())[0]
            chain_ids = list(logical_dist[example_key].keys())
            overlaps = [logical_dist[example_key][cid] for cid in chain_ids]
            
            ax2.bar(chain_ids, overlaps, alpha=0.7, color='coral')
            ax2.set_xlabel('Chain ID', fontsize=12)
            ax2.set_ylabel('Qubits in Logical Operator', fontsize=12)
            ax2.set_title(f'Logical Operator Distribution\n(Example: {example_key})', fontsize=12)
            ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

