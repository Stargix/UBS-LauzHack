import streamlit as st
import networkx as nx
from pyvis.network import Network
import pandas as pd

# Cargar el grafo desde el archivo
# Cargar los datos del archivo CSV
data = pd.read_csv("./merged_data_cleaned.csv")

G = nx.read_graphml("./Graphs/cluster_7080_graph.graphml")
# Renderizar el grafo en Streamlit
def display_graph_pyvis(G):
    net = Network(height="500px", width="100%", notebook=False)
    net.from_nx(G)
    net.show_buttons(filter_=["physics"])
    
    # Customize the appearance of the nodes and edges
    for node in net.nodes:
        node['size'] = 10
        node['color'] = '#00ccff'
    for edge in net.edges:
        edge['width'] = 1
        edge['color'] = '#cccccc'
    
    temp_path = "graph.html"
    net.save_graph(temp_path)
    return temp_path



st.title("Graph User Visualization")
graph_html = display_graph_pyvis(G)
with open(graph_html, "r") as f:
    html_content = f.read()
st.components.v1.html(html_content, height=550, scrolling=True)
