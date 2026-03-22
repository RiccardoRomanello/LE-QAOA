from abc import ABC, abstractmethod
from framework.max2sat import Max2SATFormula

class BaseConstrainedProblem(ABC):
    """
    Abstract Base Class for constrained optimization problems.
    Users must implement the reduction to Max-2SAT and repair strategies.
    """
    def __init__(self, instance_data):
        self.instance = instance_data

    @abstractmethod
    def to_max2sat(self) -> Max2SATFormula:
        """
        Performs the L-reduction from the problem domain to Max-2SAT logic.
        Should return a Max2SATFormula instance.
        """
        pass

    @abstractmethod
    def get_repair_strategies(self):
        """
        Returns a dictionary of classical repair heuristics: {name: function}.
        Each function should take a bitstring and return a valid one.
        """
        pass

    @abstractmethod
    def evaluate(self, bitstring):
        """
        Evaluates the objective function for a valid solution (e.g., set size).
        """
        pass

    def get_direct_hamiltonians(self):
        """
        Optional: Returns (H_cost, H_mixer) for a native QAOA approach.
        Used for benchmarking purposes. Returns None by default.
        """
        return None