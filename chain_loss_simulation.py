"""
Chain Loss Correction Simulation for Trapped Ion Quantum Computers

This module implements the chain loss correction scheme from:
"Correction of chain losses in trapped ion quantum computers"
by Coble, Ye, and Delfosse (arXiv:2511.16632)

Key components:
- Distributed quantum error correction codes over multiple chains
- Beacon qubits for chain loss detection
- Decoder handling circuit faults and erasures
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum
import random


class QubitState(Enum):
    """Qubit state representation"""
    ZERO = 0
    ONE = 1
    ERASED = 2  # Maximally mixed state after chain loss
    LOST = 3    # Chain is lost, qubit doesn't exist


@dataclass
class Chain:
    """Represents a single ion chain"""
    chain_id: int
    data_qubits: List[int]  # Indices of data qubits in this chain
    ancilla_qubits: List[int]  # Indices of ancilla qubits
    beacon_qubits: List[int]  # Indices of beacon qubits
    is_lost: bool = False
    loss_time: Optional[int] = None
    
    def get_all_qubits(self) -> List[int]:
        """Get all qubit indices in this chain"""
        return self.data_qubits + self.ancilla_qubits + self.beacon_qubits


@dataclass
class CircuitOperation:
    """Represents a quantum circuit operation"""
    operation_type: str  # 'gate', 'measure', 'prepare', 'reset'
    qubits: List[int]
    time_step: int
    error_rate: float = 0.0
    
    def apply_error(self, rng: np.random.Generator) -> bool:
        """Returns True if an error occurred"""
        return rng.random() < self.error_rate


class ChainLossSimulator:
    """
    Simulator for chain loss correction in trapped ion quantum computers.
    
    Implements the protocol from the paper:
    1. Distributed codes over multiple chains
    2. Beacon qubits for loss detection
    3. Erasure conversion and decoding
    """
    
    def __init__(
        self,
        num_chains: int,
        qubits_per_chain: int,
        data_qubits_per_chain: int,
        ancilla_qubits_per_chain: int,
        beacon_qubits_per_chain: int = 1,
        two_qubit_gate_error_rate: float = 1e-3,
        measurement_error_rate: float = 1e-3,
        chain_loss_rate: float = None,
        alpha: float = 1.9,
        rng_seed: Optional[int] = None
    ):
        """
        Initialize the simulator.
        
        Args:
            num_chains: Number of ion chains
            qubits_per_chain: Total qubits per chain
            data_qubits_per_chain: Data qubits per chain
            ancilla_qubits_per_chain: Ancilla qubits per chain
            beacon_qubits_per_chain: Beacon qubits per chain (default 1)
            two_qubit_gate_error_rate: Physical 2-qubit gate error rate p
            measurement_error_rate: Measurement error rate
            chain_loss_rate: Direct chain loss rate (if None, uses p^alpha)
            alpha: Exponent for chain loss rate (p_loss = p^alpha)
            rng_seed: Random seed for reproducibility
        """
        self.num_chains = num_chains
        self.qubits_per_chain = qubits_per_chain
        self.data_qubits_per_chain = data_qubits_per_chain
        self.ancilla_qubits_per_chain = ancilla_qubits_per_chain
        self.beacon_qubits_per_chain = beacon_qubits_per_chain
        
        self.two_qubit_gate_error_rate = two_qubit_gate_error_rate
        self.measurement_error_rate = measurement_error_rate
        
        if chain_loss_rate is None:
            self.chain_loss_rate = two_qubit_gate_error_rate ** alpha
        else:
            self.chain_loss_rate = chain_loss_rate
            self.alpha = alpha if chain_loss_rate is None else None
        
        self.rng = np.random.default_rng(rng_seed)
        
        # Initialize chains
        self.chains: List[Chain] = []
        self._initialize_chains()
        
        # Track qubit states
        self.qubit_states: Dict[int, QubitState] = {}
        self._initialize_qubit_states()
        
        # Statistics
        self.chain_losses: List[Tuple[int, int]] = []  # (chain_id, time_step)
        self.beacon_detections: List[Tuple[int, int, bool]] = []  # (chain_id, time, detected)
        self.operations_log: List[CircuitOperation] = []
        
    def _initialize_chains(self):
        """Initialize all chains with their qubits"""
        qubit_counter = 0
        
        for chain_id in range(self.num_chains):
            # Assign qubit indices
            data_start = chain_id * self.qubits_per_chain
            data_qubits = list(range(
                data_start,
                data_start + self.data_qubits_per_chain
            ))
            
            ancilla_start = data_start + self.data_qubits_per_chain
            ancilla_qubits = list(range(
                ancilla_start,
                ancilla_start + self.ancilla_qubits_per_chain
            ))
            
            beacon_start = ancilla_start + self.ancilla_qubits_per_chain
            beacon_qubits = list(range(
                beacon_start,
                beacon_start + self.beacon_qubits_per_chain
            ))
            
            chain = Chain(
                chain_id=chain_id,
                data_qubits=data_qubits,
                ancilla_qubits=ancilla_qubits,
                beacon_qubits=beacon_qubits
            )
            self.chains.append(chain)
            qubit_counter += self.qubits_per_chain
            
    def _initialize_qubit_states(self):
        """Initialize all qubit states"""
        for chain in self.chains:
            # Data qubits start in |0⟩
            for q in chain.data_qubits:
                self.qubit_states[q] = QubitState.ZERO
            # Ancilla qubits start in |0⟩
            for q in chain.ancilla_qubits:
                self.qubit_states[q] = QubitState.ZERO
            # Beacon qubits start in |1⟩
            for q in chain.beacon_qubits:
                self.qubit_states[q] = QubitState.ONE
                
    def simulate_chain_loss(self, chain_id: int, time_step: int) -> bool:
        """
        Simulate a chain loss event.
        
        Returns True if chain loss occurred at this time step.
        """
        # Check if chain is already lost
        if self.chains[chain_id].is_lost:
            return False
            
        # Sample chain loss probability
        loss_occurred = self.rng.random() < self.chain_loss_rate
        
        if loss_occurred:
            chain = self.chains[chain_id]
            chain.is_lost = True
            chain.loss_time = time_step
            
            # Mark all qubits in chain as erased (maximally mixed state)
            for q in chain.get_all_qubits():
                if q in self.qubit_states:
                    self.qubit_states[q] = QubitState.ERASED
                    
            self.chain_losses.append((chain_id, time_step))
            return True
            
        return False
    
    def measure_beacon(self, chain_id: int, time_step: int) -> Tuple[bool, bool]:
        """
        Measure a beacon qubit to detect chain loss.
        
        Returns: (measurement_outcome, chain_lost_detected)
        - measurement_outcome: True if measured |1⟩, False if |0⟩
        - chain_lost_detected: True if chain loss is detected
        """
        chain = self.chains[chain_id]
        
        # If chain is lost, all measurements return 0
        if chain.is_lost:
            self.beacon_detections.append((chain_id, time_step, True))
            return (False, True)
        
        # Beacon qubits should be in |1⟩
        # Measurement with possible error
        beacon_qubit = chain.beacon_qubits[0]
        true_state = self.qubit_states.get(beacon_qubit, QubitState.ONE)
        
        # Ideal measurement: should return 1 if state is ONE
        if true_state == QubitState.ONE:
            ideal_outcome = True
        else:
            ideal_outcome = False
        
        # Apply measurement error
        if self.rng.random() < self.measurement_error_rate:
            measurement_outcome = not ideal_outcome
        else:
            measurement_outcome = ideal_outcome
        
        # Chain loss detected if beacon measures 0 when it should be 1
        # (accounting for measurement errors)
        chain_lost = not measurement_outcome if true_state == QubitState.ONE else False
        
        detected = chain_lost and chain.is_lost
        self.beacon_detections.append((chain_id, time_step, detected))
        
        return (measurement_outcome, chain_lost)
    
    def measure_beacons_all_chains(self, time_step: int) -> Dict[int, bool]:
        """
        Measure beacon qubits in all chains.
        
        Returns: Dictionary mapping chain_id to whether loss was detected
        """
        detections = {}
        
        for chain_id in range(self.num_chains):
            _, detected = self.measure_beacon(chain_id, time_step)
            detections[chain_id] = detected
            
        return detections
    
    def apply_two_qubit_gate(
        self,
        qubit1: int,
        qubit2: int,
        time_step: int,
        gate_type: str = "CZ"
    ) -> bool:
        """
        Apply a two-qubit gate with possible errors.
        
        Returns True if an error occurred.
        """
        # Check if either qubit is erased/lost
        state1 = self.qubit_states.get(qubit1)
        state2 = self.qubit_states.get(qubit2)
        
        if state1 in [QubitState.ERASED, QubitState.LOST] or \
           state2 in [QubitState.ERASED, QubitState.LOST]:
            # Gate cannot be applied properly on erased qubits
            return True
        
        # Sample gate error
        error_occurred = self.rng.random() < self.two_qubit_gate_error_rate
        
        if error_occurred:
            # Apply Pauli error (simplified model)
            error_type = self.rng.choice(['X', 'Y', 'Z'])
            # In a full implementation, we would track the error syndrome
            pass
        
        op = CircuitOperation(
            operation_type='gate',
            qubits=[qubit1, qubit2],
            time_step=time_step,
            error_rate=self.two_qubit_gate_error_rate
        )
        self.operations_log.append(op)
        
        return error_occurred
    
    def measure_qubit(
        self,
        qubit: int,
        time_step: int,
        basis: str = "Z"
    ) -> Tuple[Optional[bool], bool]:
        """
        Measure a qubit.
        
        Returns: (measurement_outcome, is_erased)
        - measurement_outcome: True for |1⟩, False for |0⟩, None if erased
        - is_erased: True if qubit is erased
        """
        state = self.qubit_states.get(qubit)
        
        if state in [QubitState.ERASED, QubitState.LOST]:
            return (None, True)
        
        # Ideal measurement outcome
        ideal_outcome = (state == QubitState.ONE)
        
        # Apply measurement error
        if self.rng.random() < self.measurement_error_rate:
            measurement_outcome = not ideal_outcome
        else:
            measurement_outcome = ideal_outcome
        
        op = CircuitOperation(
            operation_type='measure',
            qubits=[qubit],
            time_step=time_step,
            error_rate=self.measurement_error_rate
        )
        self.operations_log.append(op)
        
        return (measurement_outcome, False)
    
    def replace_lost_chain(self, chain_id: int, time_step: int):
        """
        Replace a lost chain with fresh qubits in maximally mixed state.
        This converts the chain loss into an erasure.
        """
        chain = self.chains[chain_id]
        
        if not chain.is_lost:
            return
        
        # Re-initialize chain (in practice, would reload from reservoir)
        chain.is_lost = False
        chain.loss_time = None
        
        # Fresh qubits are in maximally mixed state (erased)
        for q in chain.get_all_qubits():
            # Keep as erased - decoder will handle this
            self.qubit_states[q] = QubitState.ERASED
    
    def get_erased_qubits(self) -> Set[int]:
        """Get set of all erased qubit indices"""
        erased = set()
        for qubit, state in self.qubit_states.items():
            if state == QubitState.ERASED:
                erased.add(qubit)
        return erased
    
    def get_statistics(self) -> Dict:
        """Get simulation statistics"""
        total_losses = len(self.chain_losses)
        total_operations = len(self.operations_log)
        erased_qubits = len(self.get_erased_qubits())
        
        return {
            'total_chain_losses': total_losses,
            'total_operations': total_operations,
            'erased_qubits': erased_qubits,
            'chain_loss_rate': self.chain_loss_rate,
            'gate_error_rate': self.two_qubit_gate_error_rate,
            'chain_losses': self.chain_losses,
            'beacon_detections': len(self.beacon_detections)
        }

