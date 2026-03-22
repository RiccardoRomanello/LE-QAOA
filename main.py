from framework.qaoa_engine import QAOAEngine
from problems.mis_problem import MISProblem
from test_gen.graph_generator import generate_graphs

def reproduce_small_sample():
    print("Checking LE-QAOA framework integrity...")
    # Run a tiny test (N=6) to ensure the framework is linked correctly
    graphs = generate_graphs(num_graphs=1, num_nodes=6)
    problem = MISProblem(graphs[0])
    engine = QAOAEngine(p=1, shots=500, default_style="normalized")
    
    result = engine.run_experiment(problem)
    print(f"Success! Best Score found: {result['best_score']}")
    print("To reproduce full paper results, please use 'Benchmark.ipynb'.")

if __name__ == "__main__":
    reproduce_small_sample()