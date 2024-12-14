import networkx as nx
import matplotlib.pyplot as plt

def weightToRedGreen(w):
    if w < 0:
        colour = (w*-1.0, 0, 0)
    else:
        colour = (0, w, 0)
    return colour

def drawPhenotype(genome, labels=None, node_size=1000, node_color='lightblue', arrow_size=20, font_size=10):
    layers: dict = genome[0]
    edges = genome[1]

    G = nx.DiGraph()
    G.add_nodes_from(layers)
    for node in layers:
        G.nodes[node]['layer'] = layers[node]
    for start, end, value in edges:
        G.add_edge(start, end, color=weightToRedGreen(value), weight=value)
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    edge_labels = {(u, v): f'{d['weight']:.2f}' 
               for (u, v, d) in G.edges(data=True)}

    pos = nx.multipartite_layout(G, 
                            subset_key='layer',  # Node attribute for layer
                            align='vertical')  # 'vertical' for top-to-bottom
     # Create figure and axis
    plt.figure(figsize=(10, 8), facecolor='gray')
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size)
    nx.draw_networkx_edges(G, pos, arrowsize=arrow_size, edge_color=edge_colors, width=3)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.1)
    
    # Add labels if provided, otherwise use node names
    if labels is None:
        labels = {node: str(node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=font_size)
    
    plt.axis('off')
    plt.tight_layout()
    return plt

