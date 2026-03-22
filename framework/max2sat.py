import pennylane as qml
import numpy as np

class Max2SATFormula:
    """
    Represents a Maximum 2-Satisfiability (Max-2SAT) formula.
    Provides methods to convert logical clauses into Quantum Hamiltonians.
    """
    def __init__(self, num_vars):
        self.num_vars = num_vars
        self.clauses = [] # Format: (weight, [(var_id, is_positive), ...])

    def add_clause(self, weight, literals):
        """Adds a clause with 1 or 2 literals to the formula."""
        self.clauses.append((weight, literals))

    def to_hamiltonian(self, style="standard"):
        """
        Translates the logical formula into a PennyLane Hamiltonian.
        
        Supported styles (translated from qubo_clauses.py):
        - 'standard': Full Ising mapping including the identity constant.
        - 'shifted': Removes the identity term for a leaner Hamiltonian.
        - 'normalized': Scales coefficients by the total absolute weight of the clause.
        """
        all_coeffs = []
        all_ops = []

        for weight, literals in self.clauses:
            # p_i: +1 for positive literal (x), -1 for negative literal (NOT x)
            p = [1 if lit[1] else -1 for lit in literals]
            vars = [lit[0] for lit in literals]
            
            if len(literals) == 1:
                # 1-variable clause penalty: weight/2 * (1 + p1*Z1)
                t_id, t_z1 = weight/2, (weight * p[0])/2
                coeffs, ops = self._apply_encoding_style(t_id, t_z1, 0, 0, vars, style, size=1)
            
            elif len(literals) == 2:
                # 2-variable clause penalty: weight/4 * (1 + p1*Z1 + p2*Z2 + p1*p2*Z1*Z2)
                t_id = weight/4
                t_z1 = (weight * p[0])/4
                t_z2 = (weight * p[1])/4
                t_zz = (weight * p[0] * p[1])/4
                coeffs, ops = self._apply_encoding_style(t_id, t_z1, t_z2, t_zz, vars, style, size=2)
            
            all_coeffs.extend(coeffs)
            all_ops.extend(ops)

        # Build the final Hamiltonian object in one call to avoid validation errors
        return qml.Hamiltonian(all_coeffs, all_ops)

    def _apply_encoding_style(self, t_id, t_z1, t_z2, t_zz, vars, style, size):
        """Internal helper to apply specific QUBO/Ising mathematical transformations."""
        if size == 1:
            if style == "shifted": 
                return [t_z1], [qml.PauliZ(vars[0])]
            if style == "normalized":
                norm = abs(t_id) + abs(t_z1)
                return [t_id/norm, t_z1/norm], [qml.Identity(vars[0]), qml.PauliZ(vars[0])]
            # Default: Standard/Ising
            return [t_id, t_z1], [qml.Identity(vars[0]), qml.PauliZ(vars[0])]
        
        else: # size == 2
            if style == "shifted":
                return ([t_z1, t_z2, t_zz], 
                        [qml.PauliZ(vars[0]), qml.PauliZ(vars[1]), qml.PauliZ(vars[0]) @ qml.PauliZ(vars[1])])
            if style == "normalized":
                norm = abs(t_id) + abs(t_z1) + abs(t_z2) + abs(t_zz)
                return ([t_id/norm, t_z1/norm, t_z2/norm, t_zz/norm], 
                        [qml.Identity(vars[0]), qml.PauliZ(vars[0]), qml.PauliZ(vars[1]), qml.PauliZ(vars[0]) @ qml.PauliZ(vars[1])])
            # Default: Standard/Ising
            return ([t_id, t_z1, t_z2, t_zz], 
                    [qml.Identity(vars[0]), qml.PauliZ(vars[0]), qml.PauliZ(vars[1]), qml.PauliZ(vars[0]) @ qml.PauliZ(vars[1])])