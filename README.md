# LE-QAOA: A Complexity Theory Enhanced Tool for Quantum Optimization

This repository contains the official implementation of **LE-QAOA**, a framework that uses L-reductions to solve constrained optimization problems (like Maximum Independent Set) by mapping them to unconstrained Max-2SAT.

## 📁 Repository Structure
- `framework/`: Core logic for Max-2SAT mapping and QAOA execution.
- `problems/`: Definition of the MIS problem and its repair heuristics.
- `benchmark_assets/`: Contains the specific graph instances and CSV results used in the paper.
- `Benchmark.ipynb`: The primary notebook for reproducing the plots and analysis.

## 🛠️ Setup
1. **Clone the repo:**
   `git clone` the repo (either via command line or by downloading the sources)

2. **Install the requirements:**
   ```bash
    cd LE-QAOA
    pip install -r requirements.txt
    ```

3. **Installation checks:**
   Run 
   ```bash
    python main.py
   ```
   to verify that the procedure was carried out correctly.