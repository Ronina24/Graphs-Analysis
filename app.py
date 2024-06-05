from flask import Flask, jsonify, request
import networkx as nx
from flask_cors import CORS
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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
    print(G.nodes)
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


# @app.route("/nosimilarity",methods=['POST'])
# def similarity():
#     data = request.get_json()
#     print("***********")
#     print(data)
#     if not data or 'node_stats' not in data:
#         return jsonify({"error": "Invalid input data"})

#     node_stats = data['node_stats']
#     selected_features = data['selected_features']

#     # Extract the feature vectors based on the selected features
#     feature_vectors = []
#     node_ids = []

#     for node in node_stats:
#         node_ids.append(node['id'])
#         feature_vector = []
#         for feature in selected_features:
#             feature_vector.append(node.get(feature, 0))
#         feature_vectors.append(feature_vector)

#     feature_matrix = np.array(feature_vectors)

#     # Compute cosine similarity matrix
#     similarity_matrix = cosine_similarity(feature_matrix)

#     # Prepare similarity results
#     similarity_results = []
#     for i, node1 in enumerate(node_ids):
#         for j, node2 in enumerate(node_ids):
#             if i < j:  # Avoid duplicate pairs
#                 similarity_results.append({
#                     "node1": node1,
#                     "node2": node2,
#                     "similarity": round(similarity_matrix[i, j], 10)
#                 })
    
#     print(jsonify({"similarity": similarity_results}))
#     return jsonify({"similarity": similarity_results})


def calculate_adjacency_matrix(node_stats):
    # Construct the graph
    G = nx.DiGraph()
    for node_data in node_stats:
        G.add_node(node_data['id'])

    # Add edges to the graph
    for node_data in node_stats:
        node_id = node_data['id']
        for edge_id in node_data.get('edges', []):
            G.add_edge(node_id, edge_id)

    # Compute adjacency matrix
    adj_matrix = nx.adjacency_matrix(G).toarray()
    return adj_matrix

@app.route("/similarity",methods=['POST'])
def similarity():
    data = {
    "node_stats": [
        {
            "id": "tooli0507",
            "label": "tooli0507",
            "title": "tooli0507",
            "edges": ["yosishahbar", "Bry555555", "atikt73", "naftali90"]
        },
        {
            "id": "yosishahbar",
            "label": "yosishahbar",
            "title": "yosishahbar",
            "edges": ["tooli0507", "Bry555555", "atikt73", "naftali90"]
        },
        {
            "id": "Bry555555",
            "label": "Bry555555",
            "title": "Bry555555",
            "edges": ["tooli0507", "yosishahbar", "atikt73", "naftali90"]
        },
        {
            "id": "atikt73",
            "label": "atikt73",
            "title": "atikt73",
            "edges": ["tooli0507", "yosishahbar", "Bry555555", "naftali90"]
        },
        {
            "id": "naftali90",
            "label": "naftali90",
            "title": "naftali90",
            "edges": ["tooli0507", "yosishahbar", "Bry555555", "atikt73"]
        },
        {
            "id": "es85134",
            "label": "es85134",
            "title": "es85134",
            "edges": ["bond32722733", "yosi_shahbar", "prydmn6", "demokratya11"]
        },
        {
            "id": "bond32722733",
            "label": "bond32722733",
            "title": "bond32722733",
            "edges": ["es85134", "yosi_shahbar", "prydmn6", "demokratya11"]
        },
        {
            "id": "yosi_shahbar",
            "label": "yosi_shahbar",
            "title": "yosi_shahbar",
            "edges": ["es85134", "bond32722733", "prydmn6", "demokratya11"]
        },
        {
            "id": "prydmn6",
            "label": "prydmn6",
            "title": "prydmn6",
            "edges": ["es85134", "bond32722733", "yosi_shahbar", "demokratya11"]
        },
        {
            "id": "demokratya11",
            "label": "demokratya11",
            "title": "demokratya11",
            "edges": ["es85134", "bond32722733", "yosi_shahbar", "prydmn6"]
        }
    ],
    "selected_features": []
    }

    print("***********")
    print(data)
    if not data or 'node_stats' not in data:
        return jsonify({"error": "Invalid input data"})

    node_stats = data['node_stats']
    selected_features = data['selected_features']

    # Extract the feature vectors based on the selected features
    if not selected_features:
        # If no features are selected, calculate cosine similarity based on adjacency matrix
        adj_matrix = calculate_adjacency_matrix(node_stats)
        similarity_matrix = cosine_similarity(adj_matrix)
    else:
        # Extract feature vectors and compute cosine similarity
        feature_matrix, node_ids = extract_feature_matrix(node_stats, selected_features)
        similarity_matrix = cosine_similarity(feature_matrix)

    # Prepare similarity results
    similarity_results = []
    n = len(node_stats)
    for i in range(n):
        for j in range(i + 1, n):
            similarity_results.append({
                "node1": node_stats[i]['id'],
                "node2": node_stats[j]['id'],
                "similarity": round(similarity_matrix[i, j], 10)
            })

    return jsonify({"similarity": similarity_results})

def extract_feature_matrix(node_stats, selected_features):
    feature_vectors = []
    node_ids = []
    for node in node_stats:
        node_ids.append(node['id'])
        feature_vector = [node.get(feature, 0) for feature in selected_features]
        feature_vectors.append(feature_vector)

    feature_matrix = np.array(feature_vectors)
    return feature_matrix, node_ids


def prepare_similarity_results(similarity_matrix, node_ids):
    similarity_results = []
    n = len(node_ids)
    for i in range(n):
        for j in range(i + 1, n):
            similarity_results.append({
                "node1": node_ids[i],
                "node2": node_ids[j],
                "similarity": round(similarity_matrix[i, j], 10)
            })
    return similarity_results
    
if __name__ == "__main__":
    app.run()
