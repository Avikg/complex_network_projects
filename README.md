# Complex Networks Assignments: 2023-24

## Overview
This repository contains the solutions for two assignments from the **Complex Networks (CS60078)** course. Both assignments use the **CORA dataset**, a widely-used benchmark in the graph learning community, and focus on various graph-based analysis and classification tasks.

---

## Assignment 1: Calculating Centrality Measures
### Objective
Calculate the following centrality measures for the CORA graph:
1. **Closeness Centrality**
2. **Betweenness Centrality**
3. **PageRank**

### Files and Outputs
- **Script**: `gen_centrality.py`
- **Outputs**:
  - `centralities/closeness.txt`
  - `centralities/betweenness.txt`
  - `centralities/pagerank.txt`

Each output file contains `<nodeID> <centrality_value>` pairs, sorted by descending centrality values.

### Requirements
- Python 3.8.10
- Libraries: `numpy`, `pandas`, `os`, `collections`, `sys`

### Instructions
1. Place the CORA dataset files in the `cora` folder.
2. Run the script:
   ```bash
   python3 gen_centrality.py
   ```

### Notes
- The program takes approximately **3 hours and 30 minutes** to execute.
- Ensure adequate resources (minimum 2GB RAM).

---

## Assignment 2: Node Classification
### Objective
Perform node classification using two approaches:
1. **Logistic Regression (LR)** with Node2Vec-generated embeddings.
2. **Graph Convolutional Networks (GCN)** based on Kipf's ICLR 2017 paper.

### Files and Outputs
#### Part 1: Node2Vec and Logistic Regression
- **Script**: `LR/LR.py`
- **Output**: `LR/lr_metrics.txt`

#### Part 2: Graph Convolutional Networks (GCN)
- **Script**: `GCN/GCN.py`
- **Output**: `GCN/gcn_metrics.txt`

### Requirements
- Python 3.8.10
- Libraries: `numpy`, `pandas`, `networkx`, `sklearn`, `torch`, `torch_geometric`

### Instructions
1. Place the CORA dataset files in the `data` folder.
2. Run the scripts:
   - Logistic Regression:
     ```bash
     python3 LR/LR.py
     ```
   - GCN:
     ```bash
     python3 GCN/GCN.py
     ```

### Comparison and Insights
- Detailed results and insights are provided in `Analysis.pdf`.
- Key metrics:
  - **Logistic Regression**: Accuracy = 0.8512, F1 Score = 0.8469
  - **GCN**: Accuracy = 0.9032, F1 Score = 0.9031

---

## General Notes
- Ensure all dataset files are correctly placed as instructed in each assignment.
- Install missing libraries using:
  ```bash
  pip3 install <library_name>
  ```
- Report any long runtimes or convergence issues in the `instructions.txt` file.

---

## Credits
- **Name**: Avik Pramanick
- **Roll Number**: 23CS60R78
