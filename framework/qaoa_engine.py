import pennylane as qml
from pennylane import qaoa
from pennylane import numpy as np

class QAOAEngine:
    """
    Automated Engine for benchmarking QAOA performance across different 
    Hamiltonian encodings and classical repair strategies.
    """
    def __init__(self, p=2, shots=500, default_style="standard"):
        self.p = p
        self.shots = shots
        self.default_style = default_style

    @staticmethod
    def optimize_qaoa(H_cost, H_mixer, num_qubits, p=1, steps=50, lr=0.1):
        dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(dev)
        def cost_fn(params):
            for i in range(p):
                qml.qaoa.cost_layer(params[i,0], H_cost)
                qml.qaoa.mixer_layer(params[i,1], H_mixer)
            return qml.expval(H_cost)

        params = np.random.uniform(low=0, high=np.pi, size=(p,2), requires_grad=True)
        opt = qml.AdamOptimizer(lr)

        for _ in range(steps):
            params = opt.step(cost_fn, params)

        return params
    
    @staticmethod
    def sample_qaoa(H_cost, H_mixer, num_qubits, params, p=1, shots=1000):
        dev = qml.device("default.qubit", wires=num_qubits, shots=shots)

        @qml.qnode(dev)
        def circuit():
            for i in range(p):
                qml.qaoa.cost_layer(params[i,0], H_cost)
                qml.qaoa.mixer_layer(params[i,1], H_mixer)
            return qml.sample(wires=range(num_qubits))

        return circuit()

    def run_experiment(self, problem, style=None):
        """
        Runs the QAOA for a specific encoding style and finds the 
        best result among ALL available repair techniques.
        """
        # Use provided style or fall back to default
        active_style = style if style else self.default_style
        
        print(f"\n[LE-QAOA] Starting Engine...")
        print(f"[LE-QAOA] Problem Type: {type(problem).__name__}")
        print(f"[LE-QAOA] Encoding Style: {active_style}")
        
        formula = problem.to_max2sat()
        n_qubits = formula.num_vars
        H_cost = formula.to_hamiltonian(style=active_style)
        H_mixer = qaoa.x_mixer(range(n_qubits))
        
        print(f"[LE-QAOA] Qubits: {n_qubits} | Optimization Steps: Starting...")

        params = QAOAEngine.optimize_qaoa(H_cost, H_mixer, n_qubits, p=self.p)
        samples = QAOAEngine.sample_qaoa(H_cost, H_mixer, n_qubits, params, p=self.p, shots=self.shots)
        
        print(f"[LE-QAOA] Sampling complete. Applying repair techniques...")

        repairs = problem.get_repair_strategies()
        global_best_score = -1
        best_technique_name = ""

        for name, repair_func in repairs.items():
            # Apply current repair to all samples
            current_scores = [problem.evaluate(repair_func(s.tolist())) for s in samples]
            max_for_this_technique = max(current_scores) if current_scores else 0
            
            print(f"   > Technique '{name}': Best Score = {max_for_this_technique}")

            # Update the global winner
            if max_for_this_technique > global_best_score:
                global_best_score = max_for_this_technique
                best_technique_name = name

        print(f"[LE-QAOA] Winner for {active_style}: {best_technique_name} with score {global_best_score}")
        
        return {
            "style": active_style,
            "best_technique": best_technique_name,
            "best_score": global_best_score
        }
    