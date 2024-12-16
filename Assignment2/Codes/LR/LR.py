import networkx as graph_engine
import numpy as np
from collections import defaultdict
from gensim.models import Word2Vec as VectorSpaceModel
from sklearn.linear_model import LogisticRegression as PredictionModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder as CategoricalConverter
import os

# Constructing a citation network
def build_citation_map(source_file):
    connections = {}
    with open(source_file, 'r') as src:
        for citation in src:
            recipient, issuer = map(int, citation.strip().split())
            if issuer not in connections:
                connections[issuer] = []
            if recipient not in connections:
                connections[recipient] = []
            connections[issuer].append(recipient)
    return connections

# Enhancing the network for nodes lacking citations
def enhance_for_lonely_nodes(citations):
    isolated = [node for node, cited in citations.items() if not cited]
    for solitary in isolated:
        for origin, cited in citations.items():
            if solitary in cited:
                citations[solitary] = citations.get(solitary, []) + [origin]
    return citations

# Compiling a comprehensive list of all unique identifiers
def compile_identifiers(network):
    node_aggregate = set()
    for origin, cited in network.items():
        node_aggregate.add(origin)
        node_aggregate.update(cited)
    return list(node_aggregate)

# Transforming the citation network into a graph structure
def transform_into_graph(structure):
    graph = graph_engine.DiGraph()
    for origin, cited in structure.items():
        for citation in cited:
            graph.add_edge(origin, citation)
    return graph

# Calculating transition probabilities for random walks
def derive_transition_probs(di_graph, bias_p, bias_q):
    transit_probs = defaultdict(lambda: defaultdict(list))
    for origin in di_graph.nodes():
        for neighbor in di_graph.neighbors(origin):
            probabilities = []
            for next_hop in di_graph.neighbors(neighbor):
                if origin == next_hop:
                    weight = bias_p
                elif next_hop in di_graph.neighbors(origin):
                    weight = 1
                else:
                    weight = bias_q
                probabilities.append(1 / weight)
            sum_probs = sum(probabilities)
            transit_probs[origin][neighbor] = [prob / sum_probs for prob in probabilities]
    return transit_probs

# Initiating the random walk process based on calculated probabilities
def initiate_walks(di_graph, transition_probs, num_walks, length_walk):
    paths = []
    for start in di_graph.nodes():
        for _ in range(num_walks):
            current_path = [start]
            while len(current_path) < length_walk:
                current = current_path[-1]
                next_options = list(di_graph[current])
                if not next_options:
                    break
                next_step = np.random.choice(next_options)
                current_path.append(next_step)
            paths.append(current_path)
    randomized_paths = [list(map(str, path)) for path in np.random.permutation(paths)]
    return randomized_paths

# Embedding generation with Node2Vec
def generate_embeddings(walks, dimension, context):
    model = VectorSpaceModel(walks, vector_size=dimension, window=context, min_count=0, sg=1)
    return model.wv

# Processing dataset for node classification
def process_dataset(content_file, train_file, test_file):
    node_info, train_nodes, test_nodes = {}, set(), set()
    try:
        with open(content_file, 'r') as content:
            for line in content:
                node_id, *_, label = line.strip().split()
                node_info[int(node_id)] = label
    except IOError as e:
        print(f"Error reading file: {e}")

    for file_path, node_set in [(train_file, train_nodes), (test_file, test_nodes)]:
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    _, citing_node = line.strip().split()
                    node_set.add(int(citing_node))
        except IOError as e:
            print(f"Error reading file: {e}")

    return node_info, train_nodes, test_nodes

# Main execution logic
def main():
    citation_map = build_citation_map('../cora.cites')
    updated_map = enhance_for_lonely_nodes(citation_map)
    graph = transform_into_graph(updated_map)
    transition_probs = derive_transition_probs(graph, 1, 1)
    walks = initiate_walks(graph, transition_probs, 10, 80)
    embeddings = generate_embeddings(walks, 100, 20)

    labels_map, train_set, test_set = process_dataset('../cora.content', '../cora_train.cites', '../cora_test.cites')
    encoder = CategoricalConverter()
    encoder.fit(list(labels_map.values()))

    X_train = np.array([embeddings[str(node)] for node in train_set if str(node) in embeddings])
    y_train = np.array([encoder.transform([labels_map[node]])[0] for node in train_set if node in labels_map])

    X_test = np.array([embeddings[str(node)] for node in test_set if str(node) in embeddings])
    y_test = np.array([encoder.transform([labels_map[node]])[0] for node in test_set if node in labels_map])

    classifier = PredictionModel(max_iter=1000).fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
    # Prepare the metrics content
    metrics_content = f"""Accuracy: {accuracy}
Precision: {precision}
Recall: {recall}
F1 Score: {f1}
"""


    # Define the full path for the metrics file within the LR directory
    metrics_file_path = "lr_metrics.txt"

    # Write the metrics to the specified file
    with open(metrics_file_path, 'w+') as metrics_file:
        metrics_file.write(metrics_content)

    print(f"Metrics written to {metrics_file_path}")

if __name__ == '__main__':
    main()
