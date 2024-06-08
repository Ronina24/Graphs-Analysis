from flask import Flask, jsonify, request
import networkx as nx
from flask_cors import CORS
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from waitress import serve


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
    except nx.NetworkXError as e:
        print("Error calculating eccentricity:", e)
        eccentricity = {}
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
            "closeness_centrality": format_value(closeness_centrality.get(node, None)),
            "clustering_coefficient": format_value(clustering_coefficient.get(node, None)),
            "page_rank": format_value(page_rank.get(node, None)),
            "eccentricity": format_value(eccentricity.get(node, None))
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
        'closeness_centrality': nx.closeness_centrality,
        'eigenvector_centrality': nx.eigenvector_centrality,
        'clustering_coefficient': nx.clustering,
        'pagerank': nx.pagerank
    }

    features = {}
    for node in graph.nodes():
        feature_vector = []
        for feature in selected_features:
            if feature in feature_functions:
                feature_value = feature_functions[feature](graph).get(node, 0)  # If the feature is not computed for a node, use 0
                feature_vector.append(feature_value)
        features[node] = feature_vector
    
    return features

@app.route("/similarity",methods=['POST'])
def compute_similarity():
    print("/similarity")
    data = request.get_json()
    if not data or 'selected_features' not in data:
        return jsonify({"error": "Invalid input data"}), 400
    edges = data['graphData']['edges']
    selected_features = data['selected_features']

    # Create a single graph
    original_graph = nx.DiGraph()

    for edge in edges:
        from_node = edge['from']
        to_node = edge['to']
        original_graph.add_edge(from_node, to_node)

    # Reverse the direction of edges to create the reverse graph
    reverse_graph = original_graph.reverse()

    # Extract selected features from the graph
    features = extract_selected_features(original_graph, selected_features)

    # Compute cosine similarity between nodes
    nodes = list(features.keys())
    num_nodes = len(nodes)
    similarity_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            feature_vector_i = np.array(features[nodes[i]])
            feature_vector_j = np.array(features[nodes[j]])
            similarity = cosine_similarity([feature_vector_i], [feature_vector_j])[0][0]
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Cosine similarity is symmetric

    similarity_list = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if i != j:
                similarity_list.append({'node1': nodes[i], 'node2': nodes[j], 'similarity': similarity_matrix[i, j]})
    return jsonify({'similarity_list': similarity_list})
    



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
        adj_matrix[to_node_index][from_node_index] = 1  # Since the graph is undirected, set symmetrically
    
    # Calculate cosine similarity between nodes
    cosine_sim = cosine_similarity(adj_matrix)

    # Prepare the response data
    result = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):  # Iterate over upper triangle of the matrix to avoid duplicate pairs
            result.append({
                "node1": nodes[i],
                "node2": nodes[j],
                "similarity": cosine_sim[i][j]
            })


    return jsonify({"cosine_similarity": result})

    
if __name__ == "__main__":
    serve(app.run(),port='8000')
