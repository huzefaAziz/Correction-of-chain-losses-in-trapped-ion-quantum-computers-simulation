"""
Erasure Decoder for Chain Loss Correction

Implements decoding for a combination of circuit faults and erasures
as described in the chain loss correction paper.
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass
from enum import Enum


@dataclass
class Syndrome:
    """Stabilizer syndrome measurement"""
    stabilizer_id: int
    measurement_outcome: bool  # True for +1 eigenvalue, False for -1
    time_step: int
    is_erased: bool = False  # True if measurement is unreliable due to erasure


@dataclass
class ErasurePattern:
    """Pattern of erased qubits"""
    erased_qubits: Set[int]
    time_step: int
    chain_ids: List[int]  # Chains that were lost/erased


class ErasureDecoder:
    """
    Decoder for quantum error correction with erasures.
    
    Handles a combination of:
    - Circuit faults (bit-flip, phase-flip errors)
    - Erasures (chain losses converted to maximally mixed state)
    """
    
    def __init__(
        self,
        code_stabilizers: List[List[int]],  # Support of each stabilizer
        code_checks: List[List[int]],  # Check matrix
        logical_operators: Dict[str, List[List[int]]],  # X and Z logical operators
    ):
        """
        Initialize the decoder.
        
        Args:
            code_stabilizers: List of stabilizer supports (qubit indices)
            code_checks: Parity check matrix (stabilizer-to-qubit mapping)
            logical_operators: Dictionary with 'X' and 'Z' keys mapping to logical operators
        """
        self.code_stabilizers = code_stabilizers
        self.code_checks = code_checks
        self.logical_operators = logical_operators
        self.syndromes: List[Syndrome] = []
        self.erasure_patterns: List[ErasurePattern] = []
        
    def add_syndrome(
        self,
        stabilizer_id: int,
        measurement_outcome: bool,
        time_step: int,
        is_erased: bool = False
    ):
        """Add a syndrome measurement"""
        syndrome = Syndrome(
            stabilizer_id=stabilizer_id,
            measurement_outcome=measurement_outcome,
            time_step=time_step,
            is_erased=is_erased
        )
        self.syndromes.append(syndrome)
        
    def add_erasure_pattern(
        self,
        erased_qubits: Set[int],
        time_step: int,
        chain_ids: List[int]
    ):
        """Add an erasure pattern"""
        pattern = ErasurePattern(
            erased_qubits=erased_qubits,
            time_step=time_step,
            chain_ids=chain_ids
        )
        self.erasure_patterns.append(pattern)
        
    def decode_with_erasures(
        self,
        syndromes: List[Syndrome],
        erased_qubits: Set[int]
    ) -> Tuple[List[int], bool]:
        """
        Decode syndromes with erasures.
        
        Returns:
            (correction_operators, logical_error)
            - correction_operators: List of qubit indices to apply correction
            - logical_error: True if logical error occurred
        """
        # Simplified decoder: combines standard error correction with erasure handling
        
        # 1. Filter out syndromes affected by erasures
        reliable_syndromes = [
            s for s in syndromes
            if not s.is_erased and not self._syndrome_affected_by_erasure(s, erased_qubits)
        ]
        
        # 2. Find error locations from syndromes
        error_locations = self._find_errors_from_syndromes(reliable_syndromes)
        
        # 3. Account for erasures (erased qubits contribute to syndrome)
        correction_qubits = self._merge_errors_and_erasures(error_locations, erased_qubits)
        
        # 4. Check for logical errors
        logical_error = self._check_logical_error(correction_qubits)
        
        return (correction_qubits, logical_error)
    
    def _syndrome_affected_by_erasure(
        self,
        syndrome: Syndrome,
        erased_qubits: Set[int]
    ) -> bool:
        """Check if syndrome measurement is affected by erased qubits"""
        stabilizer_support = self.code_stabilizers[syndrome.stabilizer_id]
        return bool(set(stabilizer_support) & erased_qubits)
    
    def _find_errors_from_syndromes(
        self,
        syndromes: List[Syndrome]
    ) -> Set[int]:
        """
        Find error locations from syndrome measurements.
        
        This is a simplified implementation. A full decoder would use
        more sophisticated algorithms like MWPM or belief propagation.
        """
        error_qubits = set()
        
        # Simplified: find qubits that appear in violated stabilizers
        violated_stabilizers = [
            s.stabilizer_id for s in syndromes
            if not s.measurement_outcome  # -1 eigenvalue indicates violation
        ]
        
        # Find qubits in the support of violated stabilizers
        for stab_id in violated_stabilizers:
            if stab_id < len(self.code_stabilizers):
                error_qubits.update(self.code_stabilizers[stab_id])
        
        return error_qubits
    
    def _merge_errors_and_erasures(
        self,
        error_locations: Set[int],
        erased_qubits: Set[int]
    ) -> List[int]:
        """
        Merge error corrections with erasure corrections.
        
        For erasures, we need to apply corrections that project back
        to the code space. This is simplified here.
        """
        # Combine error locations with erased qubits
        all_corrections = error_locations | erased_qubits
        return list(all_corrections)
    
    def _check_logical_error(
        self,
        correction_qubits: List[int]
    ) -> bool:
        """
        Check if the correction causes a logical error.
        
        This checks if the correction anti-commutes with any logical operator.
        """
        correction_set = set(correction_qubits)
        
        # Check X logical operators
        for logical_op in self.logical_operators.get('X', []):
            if self._commutes_with_logical(correction_set, logical_op):
                continue
            # If correction anti-commutes with logical operator, logical error
            return True
        
        # Check Z logical operators
        for logical_op in self.logical_operators.get('Z', []):
            if self._commutes_with_logical(correction_set, logical_op):
                continue
            # If correction anti-commutes with logical operator, logical error
            return True
        
        return False
    
    def _commutes_with_logical(
        self,
        correction_set: Set[int],
        logical_op: List[int]
    ) -> bool:
        """
        Check if correction commutes with a logical operator.
        
        Simplified: checks if intersection has even size (commutes)
        or odd size (anti-commutes).
        """
        intersection = correction_set & set(logical_op)
        return len(intersection) % 2 == 0
    
    def reset(self):
        """Reset decoder state"""
        self.syndromes = []
        self.erasure_patterns = []

