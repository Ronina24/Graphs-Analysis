from flask import Flask, jsonify, request
import networkx as nx
from flask_cors import CORS
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import os


app = Flask(__name__)
CORS(app) 

@app.route("/")
def home():
    return "Welcome to  Flask Web Application!"

@app.route("/stats",methods=['POST'])
def statistics():
    print("STATS")
    graphData = request.get_json()
    if not graphData['nodes'] or not graphData['edges']:
        return jsonify({"error": "Graph data is empty"})

    G = nx.DiGraph()  # Use directed graph
    for node_data in graphData['nodes']:
        G.add_node(node_data['id'], label=node_data['label'], title=node_data['title'])

    for edge_data in graphData['edges']:
        G.add_edge(edge_data['from'], edge_data['to'])

    degree_centrality = nx.degree_centrality(G)
    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    clustering_coefficient = nx.clustering(G)
    page_rank = nx.pagerank(G)

    try:
        eccentricity = nx.eccentricity(G)
    except nx.NetworkXError:
        eccentricity = {node: None for node in G.nodes}
    
    def format_value(value):
        return round(value,10) if value is not None else None
    response_data = []
    for node in G.nodes:
        
        node_data = {
            "id": node,
            "label": G.nodes[node]['label'],
            "degree_centrality": format_value(degree_centrality.get(node, None)),
            "in_degree_centrality": format_value(in_degree_centrality.get(node, None)),
            "out_degree_centrality": format_value(out_degree_centrality.get(node, None)),
            "betweenness_centrality": format_value(betweenness_centrality.get(node, None)),
            "eccentricity": format_value(eccentricity.get(node, None)),
            "closeness_centrality": format_value(closeness_centrality.get(node, None)),
            "clustering_coefficient": format_value(clustering_coefficient.get(node, None)),
            "page_rank": format_value(page_rank.get(node, None)),
        }
        response_data.append(node_data)
    
    return jsonify(response_data)


def extract_selected_features(graph, selected_features):
    print("Extracting selected features from graph")
    feature_functions = {
        'degree_centrality': nx.degree_centrality,
        'in_degree_centrality': nx.in_degree_centrality,
        'out_degree_centrality': nx.out_degree_centrality,
        'betweenness_centrality': nx.betweenness_centrality,
        'eccentricity': nx.eccentricity,
        'closeness_centrality': nx.closeness_centrality,
        'clustering_coefficient': nx.clustering,
        'pagerank': nx.pagerank
    }

    features = {}
    for node in graph.nodes():
        feature_vector = []
        for feature in selected_features:
            if feature in feature_functions:
                try:
                    feature_value = feature_functions[feature](graph).get(node, 0)
                except nx.NetworkXError:
                    feature_value = 0
                feature_vector.append(feature_value)
        features[node] = feature_vector
    
    return features

@app.route("/similarity", methods=['POST'])
def compute_similarity():
    print("/similarity")
    data = request.get_json()
    if not data or 'selected_features' not in data:
        return jsonify({"error": "Invalid input data"}), 400
    edges = data['graphData']['edges']
    selected_features = data['selected_features']

    original_graph = nx.DiGraph()
    for edge in edges:
        original_graph.add_edge(edge['from'], edge['to'])

    features = extract_selected_features(original_graph, selected_features)
    if not features:
        return jsonify({"error": "No features available for computation"}), 400

    nodes = list(features.keys())
    num_nodes = len(nodes)
    feature_vectors = np.array(list(features.values()))

    # Using scipy's cdist for efficient computation
    distance_matrix = cdist(feature_vectors, feature_vectors, metric='cosine')
    similarity_matrix = 1 - distance_matrix  # Since cosine similarity ranges from 0 to 1

    # Flatten the similarity matrix and create a list of tuples for sorting
    flattened_similarities = [(i, j, similarity_matrix[i, j]) for i in range(num_nodes) for j in range(i+1, num_nodes)]
    similarities_sorted = sorted(flattened_similarities, key=lambda x: x[2])

    # Convert sorted tuples back to dictionary format
    results = [{'node1': nodes[i], 'node2': nodes[j], 'similarity': similarity} for i, j, similarity in similarities_sorted]

    return jsonify({'similarity_list': results})


@app.route("/adjacency",methods=['POST'])
def adjacency():
    print("/adjacency")
    G = nx.Graph()
    data = request.get_json()
    edges = data['edges']
    for edge in edges:
        G.add_edge(edge['from'], edge['to'])
    
    # Get the nodes in the graph and their indices
    nodes = list(G.nodes())
    node_indices = {node: i for i, node in enumerate(nodes)}
    
    # Initialize an empty adjacency matrix
    adj_matrix = np.zeros((len(nodes), len(nodes)))
    
    # Fill the adjacency matrix based on the graph structure
    for edge in edges:
        from_node_index = node_indices[edge['from']]
        to_node_index = node_indices[edge['to']]
        adj_matrix[from_node_index][to_node_index] = 1

    # Calculate cosine similarity between nodes
    cosine_sim = cosine_similarity(adj_matrix)

    # Prepare the response data
    result = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)): 
            similarity_value = cosine_sim[i][j]
            if similarity_value != 0:
                result.append({
                    "node1": nodes[i],
                    "node2": nodes[j],
                    "similarity": cosine_sim[i][j]
                })

    similarity_list = sorted(result, key=lambda x: x['similarity'])[:500]
    return jsonify({"cosine_similarity": similarity_list})

    
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
