import networkx as nx
import matplotlib.pyplot as plt
import rtNeatDataClasses as dc

def weightToRedGreen(w):
    if w < 0:
        colour = (w*-1.0, 0, 0)
    else:
        colour = (0, w, 0)
    return colour

def drawPhenotype(genome:dc.Genome):
    #need a list of nodes (just numbers), a list of edges (tuples to go between nodes), and weights (to match with edges)
    

    pass

def draw_directed_graph(edges, weights, layers, labels=None, node_size=1000, node_color='lightblue', 
                       arrow_size=20, font_size=10):
    """
    Draw a directed graph given a list of edges.
    
    Args:
    edges: List of tuples representing edges (from_node, to_node)
    labels: Optional dictionary of node labels {node: label}
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add edges to the graph
    for i, (start, end) in enumerate(edges):
        G.add_edge(start, end, color=weightToRedGreen(weights[i]))

      # Set layer attributes
    for node, layer in layers.items():
        G.nodes[node]['layer'] = layer
    
    # Create the layout
    pos = nx.multipartite_layout(G, 
                            subset_key='layer',  # Node attribute for layer
                            align='vertical')  # 'vertical' for top-to-bottom
    
  
    
    # Create figure and axis
    plt.figure(figsize=(10, 8), facecolor='gray')
    colors = [G[u][v]['color'] for u, v in edges]
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size)
    nx.draw_networkx_edges(G, pos, arrowsize=arrow_size, edge_color=colors, width=3)
    
    # Add labels if provided, otherwise use node names
    if labels is None:
        labels = {node: str(node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=font_size)
    
    plt.axis('off')
    
    plt.tight_layout()
    
    return plt