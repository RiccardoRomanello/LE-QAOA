from framework.base_problem import BaseConstrainedProblem
from framework.max2sat import Max2SATFormula
import pennylane.qaoa as qaoa
import random

class MISProblem(BaseConstrainedProblem):
    """
    Maximum Independent Set (MIS) implementation using L-reduction to Max-2SAT.
    """
    def __init__(self, graph, penalty=2.0):
        super().__init__(graph)
        self.penalty = penalty

    def to_max2sat(self) -> Max2SATFormula:
        num_nodes = len(self.instance.nodes)
        formula = Max2SATFormula(num_nodes)
        
        # GOAL: Maximize selected nodes -> Clause (x_v) with weight 1.0
        for v in self.instance.nodes:
            formula.add_clause(1.0, [(v, True)])
            
        # CONSTRAINT: Adjacent nodes cannot both be 1 -> (NOT u OR NOT v)
        for u, v in self.instance.edges:
            formula.add_clause(self.penalty, [(u, False), (v, False)])
            
        return formula

    def get_repair_strategies(self):
        """Returns the set of classical heuristics for MIS repair."""
        return {
            "degree_based": lambda x: MISProblem.repair_degree(self.instance, x),
            "clause_count": lambda x: MISProblem.repair_clause_count(self.instance, x),
            "random_choice": lambda x: MISProblem.repair_random(self.instance, x),
            "energy_minim": lambda x: MISProblem.repair_energy(self.instance, x)
        }

    @staticmethod
    def independent_set_size(bitstring):
        return sum(bitstring)

    def evaluate(self, bitstring):
        """Metric: The size of the independent set."""
        return MISProblem.independent_set_size(bitstring)

    def get_direct_hamiltonians(self):
        """Provides the native PennyLane MIS Hamiltonian for comparison."""
        return qaoa.max_independent_set(self.instance)
    
    @staticmethod
    def find_violations(graph, x):
        """Finds pairs of adjacent nodes that are both selected (violating the independent set constraint)."""
        return [(u,v) for u,v in graph.edges() if x[u]==1 and x[v]==1]

    @staticmethod
    def repair_degree(graph, x):
        """Repairs the solution by removing the node with higher degree in each violating pair."""
        x = x.copy()
        while True:
            viol = MISProblem.find_violations(graph, x)
            if not viol: return x
            u,v = viol[0]
            x[u]=0 if graph.degree(u)>=graph.degree(v) else x[u]
            x[v]=0 if graph.degree(v)>graph.degree(u) else x[v]

    @staticmethod
    def repair_clause_count(graph,x):
        """Repairs the solution by removing the node that appears in more violated clauses."""
        x=x.copy()
        while True:
            viol=MISProblem.find_violations(graph,x)
            if not viol: return x
            u,v=viol[0]
            x[u]=0 if graph.degree(u)<=graph.degree(v) else x[u]
            x[v]=0 if graph.degree(v)<graph.degree(u) else x[v]

    @staticmethod
    def repair_random(graph,x):
        """Repairs the solution by randomly removing one node from each violating pair."""
        x=x.copy()
        while True:
            viol=MISProblem.find_violations(graph,x)
            if not viol: return x
            u,v=random.choice(viol)
            x[random.choice([u,v])]=0

    @staticmethod
    def repair_energy(graph,x):
        """Repairs the solution by removing the node that leads to the largest decrease in energy."""
        x=x.copy()
        while True:
            viol=MISProblem.find_violations(graph,x)
            if not viol: return x
            u,v=viol[0]
            cu=sum(x[w] for w in graph.neighbors(u))
            cv=sum(x[w] for w in graph.neighbors(v))
            x[u if cu>=cv else v]=0