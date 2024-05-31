from flask import Flask, jsonify
import networkx as nx
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

@app.route("/")
def home():
    return "Welcome to  Flask Web Application!"

@app.route("/stats")
def statistics():
    print("STATS")
    graphData = {
        'nodes': [
            { 'id': 1, 'label': "1", 'title': "node 1 tooltip text" },
            { 'id': 2, 'label': "2", 'title': "node 2 tooltip text" },
            { 'id': 3, 'label': "3", 'title': "node 3 tooltip text" },
            { 'id': 4, 'label': "4", 'title': "node 4 tooltip text" },
            { 'id': 5, 'label': "5", 'title': "node 5 tooltip text" },
            { 'id': 6, 'label': "6", 'title': "node 6 tooltip text" }
        ],
        'edges': [
            { 'from': 1, 'to': 2 },
            { 'from': 1, 'to': 3 },
            { 'from': 1, 'to': 4 },
            { 'from': 1, 'to': 5 },
            { 'from': 2, 'to': 1 },
            { 'from': 6, 'to': 1 },
            { 'from': 5, 'to': 6 }
        ]
    }

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
    eigen_centrality = nx.eigenvector_centrality(G)
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
            "eigen_centrality": format_value(eigen_centrality.get(node, None)),
            "clustering_coefficient": format_value(clustering_coefficient.get(node, None)),
            "page_rank": format_value(page_rank.get(node, None)),
            "eccentricity": format_value(eccentricity.get(node, None))
        }
        response_data.append(node_data)

    return jsonify(response_data)

@app.route("/similarity")
def codineSimilarity():
    print("HERE")

    
if __name__ == "__main__":
    app.run()
