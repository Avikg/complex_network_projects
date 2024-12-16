# Complex Networks Assignment 1: Calculating Centrality Measures

## Overview
This project involves calculating **Closeness Centrality**, **Betweenness Centrality**, and **PageRank** for the nodes in the CORA network dataset. The centrality measures are derived from the citation graph provided in the dataset.

## Dataset
The dataset used is the **CORA dataset**, available at: [https://linqs.org/datasets/#cora](https://linqs.org/datasets/#cora). The dataset consists of:
- `cora.content`: Describes papers with binary word attributes and class labels.
- `cora.cites`: Describes the citation graph.

## Files in Repository
- `gen_centrality.py`: Main script for computing centrality measures.
- `instructions.txt`: Contains steps to execute the code and assumptions made.
- Centrality results are stored in the `centralities` folder as:
  - `closeness.txt`
  - `betweenness.txt`
  - `pagerank.txt`

## Requirements
- **Python version**: 3.8.10
- **Libraries**:
  - `numpy`
  - `pandas`
  - `os`
  - `collections`
  - `sys`

Install missing libraries using:
```bash
pip3 install <library_name>
```

## Instructions
1. Clone the repository and navigate to the project root.
2. Place the `cora.cites` file in a folder named `cora` (e.g., `cora/cora.cites`).
3. Run the following command:
   ```bash
   python3 gen_centrality.py
   ```

## Output
- The results are stored in the `centralities` folder in three separate files:
  - **Closeness Centrality**: `closeness.txt`
  - **Betweenness Centrality**: `betweenness.txt`
  - **PageRank**: `pagerank.txt`
- Each file contains lines in the format: `<nodeID> <centrality_value>` (rounded to six decimal places) and sorted by descending centrality value.

## Assumptions
1. Nodes with no outgoing edges are adjusted for better graph representation.
2. Unreachable nodes in Closeness Centrality are assigned a default maximum distance (2708 nodes).
3. PageRank iterations terminate after 50 iterations or if the values converge within six decimal places.
4. Results are sorted in descending order by centrality value.

## Execution Time
The program takes approximately **3 hours and 30 minutes** to execute due to the size of the dataset and computational complexity.

## Notes
- Ensure adequate system resources (minimum 2GB RAM) for execution.
- Report long runtimes or convergence issues as mentioned in `instructions.txt`.
