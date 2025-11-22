# Chain Loss Correction for Trapped Ion Quantum Computers

Python implementation of the chain loss correction scheme from the paper:

**"Correction of chain losses in trapped ion quantum computers"**  
by Coble, Ye, and Delfosse (arXiv:2511.16632)

## Overview

This codebase implements a solution to the chain loss problem in trapped ion quantum computers, based on three key components:

1. **Distributed quantum error correction codes** over multiple long chains
2. **Beacon qubits** within each chain to detect chain losses
3. **Erasure decoder** adapted to correct a combination of circuit faults and erasures

The implementation simulates a [[72,12,6]] bivariate bicycle (BB) code distributed over 12 chains with beacon qubits for loss detection.

## Features

- **Chain Loss Simulation**: Models ion loss events in long chains
- **Beacon Qubit Detection**: Detects chain losses using beacon qubits kept in |1⟩ state
- **Erasure Conversion**: Converts chain losses to erasures (maximally mixed state)
- **Distributed BB Code**: Implements bivariate bicycle codes distributed over multiple chains
- **Erasure Decoder**: Handles combination of circuit faults and erasures
- **Visualization Tools**: Plotting utilities for analysis and results

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- numpy
- matplotlib
- seaborn (for visualization)

## Quick Start

Run the example simulation:

```bash
python example_simulation.py
```

This will:
- Initialize a [[72,12,6]] BB code distributed over 12 chains
- Run 6 rounds of syndrome extraction with chain loss simulation
- Generate plots showing logical errors, chain losses, and erased qubits

## Code Structure

### Main Modules

- **`chain_loss_simulation.py`**: Core simulation framework for chain loss correction
  - `ChainLossSimulator`: Main simulator class
  - `Chain`: Represents a single ion chain
  - `CircuitOperation`: Quantum circuit operations

- **`erasure_decoder.py`**: Decoder for circuit faults and erasures
  - `ErasureDecoder`: Handles decoding with erasures
  - `Syndrome`: Stabilizer syndrome measurements
  - `ErasurePattern`: Pattern of erased qubits

- **`bb_code.py`**: Bivariate Bicycle (BB) code implementation
  - `BBCode`: BB code structure and distribution

- **`visualization.py`**: Plotting and visualization utilities
  - Chain loss event plots
  - Logical error rate analysis
  - Erased qubits distribution
  - Beacon detection efficiency

- **`example_simulation.py`**: Example usage script with full simulation

## Usage Example

```python
from chain_loss_simulation import ChainLossSimulator
from bb_code import BBCode
from erasure_decoder import ErasureDecoder

# Initialize BB code
code = BBCode(n=72, k=12, d=6, num_chains=12, qubits_per_chain=6)

# Initialize simulator
simulator = ChainLossSimulator(
    num_chains=12,
    qubits_per_chain=6,
    data_qubits_per_chain=4,
    ancilla_qubits_per_chain=1,
    beacon_qubits_per_chain=1,
    two_qubit_gate_error_rate=2e-3,
    alpha=1.9  # p_loss = p^alpha
)

# Initialize decoder
decoder = ErasureDecoder(
    code_stabilizers=code.stabilizers,
    code_checks=[],
    logical_operators=code.logical_operators
)

# Measure beacon qubits to detect losses
detections = simulator.measure_beacons_all_chains(time_step=0)

# Process chain losses and decode
for chain_id, detected in detections.items():
    if detected:
        simulator.replace_lost_chain(chain_id, time_step=0)

erased_qubits = simulator.get_erased_qubits()
correction, logical_error = decoder.decode_with_erasures(
    decoder.syndromes,
    erased_qubits
)
```

## Key Concepts

### Chain Loss

When an ion is lost from a chain, it typically destabilizes the entire chain, causing:
- Position shift of remaining ions
- Motional heating
- Measurement misalignment (all measurements return 0)

### Beacon Qubits

Beacon qubits are kept in |1⟩ state and measured regularly. Since lost chains can't emit photons:
- If beacon measures 1: chain is present
- If beacon measures 0: chain may be lost

Multiple beacon qubits or repeated measurements reduce false positives from measurement errors.

### Erasure Conversion

Once a chain loss is detected:
1. Lost chain is replaced with fresh ions from reservoir
2. Fresh qubits initialized in maximally mixed state (erased)
3. Decoder treats erased qubits as known error locations

### Distributed Codes

To avoid permanent data loss:
- Code qubits distributed across multiple chains
- No logical operator fully contained in single chain
- Loss of one chain doesn't delete logical information

## Simulation Parameters

Key parameters (as in the paper):

- **p**: Physical 2-qubit gate error rate (e.g., 2×10⁻³)
- **p_loss**: Chain loss rate, typically p^α where α ≈ 1.9-2.1
- **Number of chains**: 12 (for [[72,12,6]] code)
- **Beacon qubits per chain**: 1-3
- **Syndrome extraction rounds**: 6

## Results

The paper reports:
- Threshold of ~2×10⁻³ for α ≥ 1.9
- Chain loss becomes dominant error source for α < 1.9
- Fast beacon measurements significantly improve performance for low p_loss

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{coble2024correction,
  title={Correction of chain losses in trapped ion quantum computers},
  author={Coble, Nolan J. and Ye, Min and Delfosse, Nicolas},
  journal={arXiv preprint arXiv:2511.16632},
  year={2024}
}
```

## License

This implementation is provided for research and educational purposes.

## Notes

This is a simplified implementation focused on demonstrating the chain loss correction scheme. A full implementation would include:

- Complete stabilizer measurement circuits
- More sophisticated decoders (MWPM, belief propagation)
- Detailed noise models (depolarizing, amplitude damping)
- Full QCCD architecture simulation
- Optimized beacon placement and measurement schedules

