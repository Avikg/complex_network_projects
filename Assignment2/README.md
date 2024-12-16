# Complex Networks Assignment 2: Node Classification

## Overview
This project focuses on **node classification** using two approaches: **Logistic Regression (LR)** with Node2Vec-generated embeddings and **Graph Convolutional Networks (GCN)**. The CORA dataset is utilized for training and evaluating the models.

## Dataset
The dataset used is the **CORA dataset**, available at: [https://linqs.org/datasets/#cora](https://linqs.org/datasets/#cora). It consists of:
- `cora.content`: Paper descriptions with binary word attributes and class labels.
- `cora.cites`: Citation graph split into train and test sets.

## Tasks
### Part 1: Node2Vec and Logistic Regression
1. Implement Node2Vec from scratch to generate node embeddings.
2. Train a Logistic Regression model on the embeddings for a 7-class classification task.
3. Evaluate using metrics: **Accuracy**, **Precision**, **Recall**, and **F1-score**.
4. Output: `lr_metrics.txt` in the `LR` folder.

### Part 2: Graph Convolutional Networks (GCN)
1. Implement a GCN architecture with:
   - Two layers with 16 units each
   - ReLU activation and 0.5 dropout
   - Adam optimizer (learning rate: 0.01)
2. Train the GCN for the same node classification task.
3. Evaluate using the same metrics as Part 1.
4. Output: `gcn_metrics.txt` in the `GCN` folder.

## Requirements
- **Python version**: 3.8.10
- **Libraries**:
  - `numpy`
  - `pandas`
  - `networkx`
  - `sklearn`
  - `torch`
  - `torch_geometric`

Install missing libraries using:
```bash
pip3 install <library_name>
```

## Instructions
1. Clone the repository and navigate to the project root.
2. Ensure the dataset files are in the appropriate directories:
   - `cora.content` and `cora.cites` files should be in the `data` folder.
3. Run the following commands for each part:
   - For Logistic Regression:
     ```bash
     python3 LR/LR.py
     ```
   - For GCN:
     ```bash
     python3 GCN/GCN.py
     ```

## Outputs
- **Logistic Regression Metrics**: Stored in `LR/lr_metrics.txt`.
- **GCN Metrics**: Stored in `GCN/gcn_metrics.txt`.
- Evaluation metrics include:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

## Comparison and Insights
- A detailed comparison of the two approaches is available in the `Analysis.pdf` file.
- Key findings:
  - **Logistic Regression**:
    - Accuracy: 0.8512
    - F1 Score: 0.8469
  - **GCN**:
    - Accuracy: 0.9032
    - F1 Score: 0.9031

## Notes
- Ensure adequate system resources (minimum 2GB RAM) for execution.
- The training process may take significant time; report any issues as described in the `instructions.txt` file.
