"""
Bivariate Bicycle (BB) Code Implementation

Implements the BB code structure for distributed quantum error correction
as described in the chain loss correction paper.

The paper uses a [[72,12,6]] BB code distributed over 12 chains.
"""

import numpy as np
from typing import List, Dict, Set, Tuple
from itertools import combinations


class BBCode:
    """
    Bivariate Bicycle (BB) Code implementation.
    
    BB codes are quantum LDPC codes that are well-suited for
    distributed architectures like trapped ion chains.
    """
    
    def __init__(
        self,
        n: int = 72,  # Block length
        k: int = 12,  # Logical qubits
        d: int = 6,   # Distance
        num_chains: int = 12,
        qubits_per_chain: int = 6
    ):
        """
        Initialize BB code.
        
        Args:
            n: Total number of physical qubits
            k: Number of logical qubits
            d: Code distance
            num_chains: Number of chains to distribute code over
            qubits_per_chain: Qubits per chain
        """
        self.n = n
        self.k = k
        self.d = d
        self.num_chains = num_chains
        self.qubits_per_chain = qubits_per_chain
        
        # Generate stabilizer structure (simplified)
        self.stabilizers = self._generate_stabilizers()
        self.logical_operators = self._generate_logical_operators()
        
        # Distribute qubits over chains
        self.chain_distribution = self._distribute_over_chains()
        
        # Verify no logical operator is fully contained in a single chain
        self._verify_distribution()
        
    def _generate_stabilizers(self) -> List[List[int]]:
        """
        Generate stabilizer structure.
        
        This is a simplified implementation. A full BB code construction
        would use the bivariate polynomial formalism.
        """
        # Simplified: generate stabilizers with reasonable structure
        stabilizers = []
        
        # X-type stabilizers
        num_stabilizers = (self.n - self.k) // 2
        
        for i in range(num_stabilizers):
            # Create stabilizer with support on ~sqrt(n) qubits
            support_size = int(np.sqrt(self.n))
            qubits = list(range(
                i * support_size,
                min((i + 1) * support_size, self.n)
            ))
            if len(qubits) > 0:
                stabilizers.append(qubits)
        
        # Add more stabilizers to reach target number
        while len(stabilizers) < num_stabilizers:
            # Random stabilizer support
            support_size = np.random.randint(4, 8)
            qubits = sorted(np.random.choice(self.n, support_size, replace=False).tolist())
            stabilizers.append(qubits)
        
        return stabilizers[:num_stabilizers]
    
    def _generate_logical_operators(self) -> Dict[str, List[List[int]]]:
        """
        Generate logical operators.
        
        Returns dictionary with 'X' and 'Z' keys, each mapping to
        a list of logical operators (one per logical qubit).
        """
        logical_ops = {'X': [], 'Z': []}
        
        for i in range(self.k):
            # Simplified logical operators
            # In a real BB code, these would be carefully constructed
            logical_ops['X'].append(list(range(i * (self.n // self.k), (i + 1) * (self.n // self.k))))
            logical_ops['Z'].append(list(range(i * (self.n // self.k), (i + 1) * (self.n // self.k))))
        
        return logical_ops
    
    def _distribute_over_chains(self) -> Dict[int, List[int]]:
        """
        Distribute code qubits over chains.
        
        Returns: Dictionary mapping chain_id to list of qubit indices
        """
        distribution = {i: [] for i in range(self.num_chains)}
        
        # Distribute qubits evenly across chains
        for qubit_id in range(self.n):
            chain_id = qubit_id % self.num_chains
            distribution[chain_id].append(qubit_id)
        
        return distribution
    
    def _verify_distribution(self):
        """
        Verify that no logical operator is fully contained in a single chain.
        
        This is critical: if a logical operator is contained in one chain,
        losing that chain would permanently delete the logical information.
        """
        for chain_id, qubits in self.chain_distribution.items():
            chain_qubits = set(qubits)
            
            # Check X logical operators
            for i, logical_op in enumerate(self.logical_operators['X']):
                if set(logical_op).issubset(chain_qubits):
                    raise ValueError(
                        f"X logical operator {i} is fully contained in chain {chain_id}. "
                        "This would cause permanent data loss if chain is lost."
                    )
            
            # Check Z logical operators
            for i, logical_op in enumerate(self.logical_operators['Z']):
                if set(logical_op).issubset(chain_qubits):
                    raise ValueError(
                        f"Z logical operator {i} is fully contained in chain {chain_id}. "
                        "This would cause permanent data loss if chain is lost."
                    )
        
        print(f"[OK] Distribution verified: No logical operator fully contained in single chain")
    
    def get_qubits_in_chain(self, chain_id: int) -> List[int]:
        """Get all qubit indices in a given chain"""
        return self.chain_distribution.get(chain_id, [])
    
    def get_chain_for_qubit(self, qubit_id: int) -> int:
        """Get chain ID for a given qubit"""
        for chain_id, qubits in self.chain_distribution.items():
            if qubit_id in qubits:
                return chain_id
        raise ValueError(f"Qubit {qubit_id} not found in any chain")
    
    def get_stabilizer_support(self, stabilizer_id: int) -> List[int]:
        """Get qubit support of a stabilizer"""
        if stabilizer_id < len(self.stabilizers):
            return self.stabilizers[stabilizer_id]
        return []
    
    def check_logical_operator_distribution(self) -> Dict[str, Dict[int, int]]:
        """
        Check how logical operators are distributed across chains.
        
        Returns: Dictionary showing how many qubits of each logical operator
                 are in each chain.
        """
        result = {
            'X': {chain_id: {} for chain_id in range(self.num_chains)},
            'Z': {chain_id: {} for chain_id in range(self.num_chains)}
        }
        
        for op_type in ['X', 'Z']:
            for logical_idx, logical_op in enumerate(self.logical_operators[op_type]):
                for chain_id in range(self.num_chains):
                    chain_qubits = set(self.chain_distribution[chain_id])
                    overlap = len(set(logical_op) & chain_qubits)
                    result[op_type][chain_id][logical_idx] = overlap
        
        return result

