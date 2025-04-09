import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch

def visualize_enhanced_graph(graph, title="Enhanced Graph with Pass Probabilities"):
    """Visualize the enhanced graph structure with pass probability edges.
    
    Args:
        graph: PyTorch Geometric Data object with enhanced structure
        title: Plot title
    """
    plt.figure(figsize=(12, 10))
    
    # Create a NetworkX graph for visualization
    G = nx.DiGraph()
    
    # Extract node features, edge index, and edge attributes
    if not hasattr(graph, 'x') or not hasattr(graph, 'edge_index'):
        print("Graph does not have required attributes")
        return
    
    # Convert tensors to numpy if needed
    node_features = graph.x.numpy() if isinstance(graph.x, torch.Tensor) else graph.x
    edge_index = graph.edge_index.numpy() if isinstance(graph.edge_index, torch.Tensor) else graph.edge_index
    
    # Check if edge_attr exists
    if hasattr(graph, 'edge_attr'):
        edge_attr = graph.edge_attr.numpy() if isinstance(graph.edge_attr, torch.Tensor) else graph.edge_attr
    else:
        # Create default edge attributes if none exist
        edge_attr = np.ones((edge_index.shape[1], 1))
    
    # Add nodes to the graph
    for i in range(node_features.shape[0]):
        # Extract node position (x, y)
        pos_x, pos_y = node_features[i, 0], node_features[i, 1]
        
        # Determine node type (home, away, ball)
        node_type = node_features[i, 4] if node_features.shape[1] > 4 else 0
        
        # Add node with attributes
        G.add_node(i, pos=(pos_x, pos_y), type=node_type)
    
    # Add edges to the graph
    for j in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, j]), int(edge_index[1, j])
        weight = float(edge_attr[j, 0]) if j < edge_attr.shape[0] else 1.0
        
        # Add edge with weight attribute
        G.add_edge(src, dst, weight=weight)
    
    # Get node positions for drawing
    pos = nx.get_node_attributes(G, 'pos')
    
    # Get node types
    node_types = nx.get_node_attributes(G, 'type')
    
    # Create node lists by type
    home_nodes = [n for n, attr in node_types.items() if attr == 0]
    away_nodes = [n for n, attr in node_types.items() if attr == 1]
    ball_nodes = [n for n, attr in node_types.items() if attr == 2]
    
    # Draw the field
    field_length = 105
    field_width = 68
    plt.plot([0, 0, field_length, field_length, 0], [0, field_width, field_width, 0, 0], 'k-')
    plt.plot([field_length/2, field_length/2], [0, field_width], 'k-')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=home_nodes, node_color='blue', node_size=100, label='Home Team')
    nx.draw_networkx_nodes(G, pos, nodelist=away_nodes, node_color='red', node_size=100, label='Away Team')
    nx.draw_networkx_nodes(G, pos, nodelist=ball_nodes, node_color='black', node_size=150, label='Ball')
    
    # Draw edges with varying width based on weight
    edges = G.edges(data=True)
    
    # Separate proximity edges and pass probability edges
    proximity_edges = [(u, v) for u, v, d in edges if d['weight'] == 1.0]
    pass_prob_edges = [(u, v) for u, v, d in edges if d['weight'] != 1.0]
    
    # Draw proximity edges
    nx.draw_networkx_edges(G, pos, edgelist=proximity_edges, width=1, alpha=0.3, edge_color='gray')
    
    # Draw pass probability edges with varying width
    if pass_prob_edges:
        weights = [G[u][v]['weight'] for u, v in pass_prob_edges]
        # Normalize weights for better visualization
        max_weight = max(weights)
        normalized_weights = [3 * (w / max_weight) for w in weights]
        
        nx.draw_networkx_edges(G, pos, edgelist=pass_prob_edges, 
                              width=normalized_weights, alpha=0.7, 
                              edge_color='green', style='dashed',
                              arrows=True, arrowsize=10)
    
    plt.title(title)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.savefig('enhanced_graph_visualization.png')
    plt.close()
    
    print(f"Enhanced graph visualization saved to enhanced_graph_visualization.png")
