import networkx as nx
import numpy as np
import os
import json
from collections import Counter
import pyneuroml.writers as writers
from pyneuroml import pynml
from pyneuroml.utils import validate_neuroml2

class GraphToNeuroML:
    """
    Converts field-generated evolved graphs to NeuroML format with 
    biologically plausible cell type assignments and connection properties.
    """
    
    def __init__(self, original_network=None):
        """
        Initialize the converter with an optional original network for reference.
        
        Args:
            original_network (nx.Graph, optional): Original reference network
        """
        self.original_network = original_network
        
        # Parameters for cell type distributions
        self.cell_types = {
            'motor': 'M',      # Motor neurons (M-class)
            'sensory': 'I',    # Sensory neurons (I-class)
            'inter': 'MC'      # Interneurons (MC, MI, NSM-class)
        }
        
        # Target distribution based on C. elegans pharyngeal nervous system
        # These can be adjusted based on the original network analysis
        self.target_distribution = {
            'motor': 0.4,    # 40% motor neurons
            'sensory': 0.3,  # 30% sensory neurons  
            'inter': 0.3     # 30% interneurons
        }
        
        # Default NeuroML parameters
        self.default_params = {
            'motor': {
                'cell_type': 'motor',
                'threshold': -30.0,
                'refractory_period': 2.0,
                'cm': 1.0,
                'v_rest': -65.0
            },
            'sensory': {
                'cell_type': 'sensory',
                'threshold': -35.0,
                'refractory_period': 1.0,
                'cm': 0.8,
                'v_rest': -60.0
            },
            'inter': {
                'cell_type': 'interneuron',
                'threshold': -40.0,
                'refractory_period': 1.5,
                'cm': 1.2,
                'v_rest': -65.0
            }
        }
    
    def compute_network_metrics(self, G):
        """
        Compute network metrics for each node to help with cell type assignment
        
        Args:
            G (nx.Graph): The network graph
            
        Returns:
            dict: Dictionary of dictionaries with node metrics
        """
        metrics = {}
        
        # Compute basic centrality measures
        in_degree = dict(G.in_degree(weight='weight'))
        out_degree = dict(G.out_degree(weight='weight'))
        
        # Compute betweenness centrality (for interneuron identification)
        betweenness = nx.betweenness_centrality(G, weight='weight')
        
        # Community detection for module identification
        communities = None
        try:
            # Attempt to find communities with Louvain method
            import community as community_louvain
            communities = community_louvain.best_partition(G.to_undirected())
        except ImportError:
            # Fall back to connected components if python-louvain is not available
            communities = {}
            for i, comp in enumerate(nx.connected_components(G.to_undirected())):
                for node in comp:
                    communities[node] = i
        
        # Compute input/output ratio (important for neuron type)
        for node in G.nodes():
            in_val = in_degree.get(node, 0)
            out_val = out_degree.get(node, 0)
            
            # Avoid division by zero
            io_ratio = in_val / max(out_val, 1)
            
            metrics[node] = {
                'in_degree': in_val,
                'out_degree': out_val,
                'io_ratio': io_ratio,
                'betweenness': betweenness.get(node, 0),
                'community': communities.get(node, 0)
            }
            
        return metrics
    
    def assign_cell_types(self, G):
        """
        Assign cell types based on network topology metrics
        
        Args:
            G (nx.Graph): The network graph
            
        Returns:
            dict: Dictionary mapping node IDs to cell types
        """
        # Compute network metrics for all nodes
        metrics = self.compute_network_metrics(G)
        nodes = list(G.nodes())
        
        # If we have the original network, extract its cell type distribution
        original_distribution = self.target_distribution
        if self.original_network and nx.get_node_attributes(self.original_network, 'cell_type'):
            types = nx.get_node_attributes(self.original_network, 'cell_type')
            type_count = Counter(types.values())
            total = sum(type_count.values())
            
            # Update distribution based on original network
            if total > 0:
                original_distribution = {
                    'motor': type_count.get('motor', 0) / total,
                    'sensory': type_count.get('sensory', 0) / total,
                    'inter': type_count.get('inter', 0) / total
                }
        
        # Calculate target counts for each type
        n_nodes = len(nodes)
        target_counts = {
            t: int(round(n_nodes * original_distribution[t])) 
            for t in original_distribution
        }
        
        # Ensure we assign all nodes (adjust for rounding errors)
        total_assigned = sum(target_counts.values())
        if total_assigned < n_nodes:
            # Add the remaining nodes to the type with highest distribution
            max_type = max(original_distribution, key=original_distribution.get)
            target_counts[max_type] += (n_nodes - total_assigned)
        elif total_assigned > n_nodes:
            # Remove extras from the type with highest count
            max_type = max(target_counts, key=target_counts.get)
            target_counts[max_type] -= (total_assigned - n_nodes)
        
        # Score each node for each cell type based on metrics
        scores = {}
        for node in nodes:
            m = metrics[node]
            
            # Motor neurons: high out-degree, low in-degree (io_ratio < 1)
            motor_score = m['out_degree'] * (1.0 / max(m['io_ratio'], 0.1))
            
            # Sensory neurons: high in-degree, low out-degree (io_ratio > 1)
            sensory_score = m['in_degree'] * m['io_ratio']
            
            # Interneurons: balanced in/out and high betweenness
            inter_score = m['betweenness'] * (1.0 / abs(m['io_ratio'] - 1.0))
            
            scores[node] = {
                'motor': motor_score,
                'sensory': sensory_score,
                'inter': inter_score
            }
        
        # Assign cell types based on scores and target distribution
        assignments = {}
        remaining_nodes = set(nodes)
        
        # Sort nodes by highest score for each type
        for cell_type in ['motor', 'sensory', 'inter']:
            # Get target count for this type
            count = target_counts[cell_type]
            
            # Sort nodes by their score for this type (highest to lowest)
            type_scores = [(node, scores[node][cell_type]) for node in remaining_nodes]
            type_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Assign top-scoring nodes to this type
            for i in range(min(count, len(type_scores))):
                node, _ = type_scores[i]
                assignments[node] = cell_type
                remaining_nodes.remove(node)
        
        # Assign any remaining nodes to their highest-scoring type
        for node in remaining_nodes:
            node_scores = scores[node]
            assignments[node] = max(node_scores, key=node_scores.get)
        
        return assignments
    
    def assign_synapse_properties(self, G, node_types):
        """
        Assign synaptic properties to the connections in the graph
        
        Args:
            G (nx.Graph): The network graph
            node_types (dict): Dictionary mapping node IDs to cell types
            
        Returns:
            dict: Dictionary of dictionaries with edge properties
        """
        edge_props = {}
        
        # For each edge in the graph
        for source, target, data in G.edges(data=True):
            # Get cell types of source and target
            source_type = node_types.get(source, 'inter')
            target_type = node_types.get(target, 'inter')
            
            # Get edge weight or default to 1.0
            weight = data.get('weight', 1.0)
            
            # Determine if connection is excitatory or inhibitory
            # Typically:
            # - Sensory neurons mostly make excitatory connections
            # - Motor neurons often have inhibitory connections to other motor neurons
            # - Interneurons make both excitatory and inhibitory connections
            is_excitatory = True
            
            if source_type == 'motor' and target_type == 'motor':
                # Motor to motor is often inhibitory
                is_excitatory = False
            elif source_type == 'inter':
                # Interneurons make both types, with a bias towards excitatory
                is_excitatory = np.random.random() < 0.7
            
            # Calculate delay based on distance or other factors (simplified here)
            # In a real implementation, this could use spatial information if available
            delay = 0.5 + (1.0 - min(weight, 1.0)) * 1.5  # 0.5 to 2.0 ms
            
            # Scale the synaptic weight
            # Stronger connections in the evolved graph should be stronger in NeuroML
            scaled_weight = weight * 1.0  # Could scale by a factor based on type
            
            # Assign a unique ID to the connection
            conn_id = f"{source}_{target}"
            
            edge_props[conn_id] = {
                'source': source,
                'target': target,
                'weight': scaled_weight,
                'delay': delay,
                'is_excitatory': is_excitatory
            }
            
        return edge_props
    
    def create_neuroml_network(self, G, node_types=None):
        """
        Create a NeuroML network from a graph
        
        Args:
            G (nx.Graph): The network graph
            node_types (dict, optional): Dictionary mapping node IDs to cell types
            
        Returns:
            neuroml.Network: The NeuroML network model
        """
        # Create the NeuroML document
        nml_doc = neuroml.NeuroMLDocument(id="EvolvedNetwork")
        
        # Create a network
        net = neuroml.Network(id="network")
        nml_doc.networks.append(net)
        
        # If cell types haven't been provided, assign them
        if node_types is None:
            node_types = self.assign_cell_types(G)
        
        # Assign synapse properties
        synapse_props = self.assign_synapse_properties(G, node_types)
        
        # Create cell types (if they don't exist)
        cell_type_defs = {}
        
        for cell_type, params in self.default_params.items():
            # Create a cell type
            cell_id = f"{cell_type}_cell"
            
            # Use izhikevich cell model for now
            cell = neuroml.IzhikevichCell(
                id=cell_id,
                v0=params['v_rest'],
                thresh=params['threshold'],
                a=0.02,  # Standard Izhikevich parameters
                b=0.2,
                c=params['v_rest'],
                d=2.0,
                peak=30.0
            )
            
            nml_doc.izhikevich_cells.append(cell)
            cell_type_defs[cell_type] = cell_id
        
        # Create synapse types
        exc_syn = neuroml.ExpTwoSynapse(
            id="exc_syn",
            gbase="5nS",
            tau_decay="2ms",
            tau_rise="0.5ms",
            e_rev="0mV"
        )
        
        inh_syn = neuroml.ExpTwoSynapse(
            id="inh_syn",
            gbase="5nS",
            tau_decay="5ms",
            tau_rise="1ms",
            e_rev="-80mV"
        )
        
        nml_doc.exp_two_synapses.append(exc_syn)
        nml_doc.exp_two_synapses.append(inh_syn)
        
        # Add populations for each cell type
        populations = {}
        
        for cell_type in self.cell_types:
            # Count nodes of this type
            type_nodes = [n for n, t in node_types.items() if t == cell_type]
            pop_size = len(type_nodes)
            
            if pop_size > 0:
                # Create population
                pop_id = f"{cell_type}_population"
                pop = neuroml.Population(
                    id=pop_id,
                    component=cell_type_defs[cell_type],
                    size=pop_size
                )
                
                net.populations.append(pop)
                populations[cell_type] = {
                    'id': pop_id,
                    'nodes': type_nodes,
                    'size': pop_size
                }
        
        # Create node ID to population index mapping
        node_indices = {}
        for cell_type, pop_info in populations.items():
            for i, node_id in enumerate(pop_info['nodes']):
                node_indices[node_id] = {
                    'population': pop_info['id'],
                    'index': i
                }
        
        # Add projections (connections between populations)
        projections = {}
        
        # Iterate over all synaptic connections
        for conn_id, props in synapse_props.items():
            source_id = props['source']
            target_id = props['target']
            
            # Skip if nodes don't exist in our mapping
            if source_id not in node_indices or target_id not in node_indices:
                continue
            
            source_pop = node_indices[source_id]['population']
            source_idx = node_indices[source_id]['index']
            
            target_pop = node_indices[target_id]['population']
            target_idx = node_indices[target_id]['index']
            
            # Create projection ID: source_pop_to_target_pop
            proj_id = f"{source_pop}_to_{target_pop}"
            
            # Create the projection if it doesn't exist
            if proj_id not in projections:
                syn_type = "exc_syn" if props['is_excitatory'] else "inh_syn"
                
                proj = neuroml.Projection(
                    id=proj_id,
                    presynaptic_population=source_pop,
                    postsynaptic_population=target_pop,
                    synapse=syn_type
                )
                
                net.projections.append(proj)
                projections[proj_id] = proj
            
            # Add connection to the projection
            conn = neuroml.Connection(
                id=conn_id,
                pre_cell_id=f"{source_idx}",
                post_cell_id=f"{target_idx}",
                weight=str(props['weight']),
                delay=f"{props['delay']}ms"
            )
            projections[proj_id].connections.append(conn)
        
        return nml_doc
    
    def export_to_neuroml(self, G, output_path, node_types=None):
        """
        Export a graph to a NeuroML file
        
        Args:
            G (nx.Graph): The network graph
            output_path (str): Path to save the NeuroML file
            node_types (dict, optional): Dictionary mapping node IDs to cell types
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # If cell types haven't been provided, assign them
            if node_types is None:
                node_types = self.assign_cell_types(G)
            
            # Create the NeuroML network
            nml_doc = self.create_neuroml_network(G, node_types)
            
            # Write the NeuroML document to file
            writers.NeuroMLWriter.write(nml_doc, output_path)
            
            # Validate the NeuroML document
            validate_neuroml2(output_path)
            
            print(f"Successfully exported to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting to NeuroML: {e}")
            return False
    
    def export_cell_types_to_json(self, node_types, output_path):
        """
        Export the assigned cell types to a JSON file
        
        Args:
            node_types (dict): Dictionary mapping node IDs to cell types
            output_path (str): Path to save the JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert to a format suitable for JSON
            output_data = {
                'node_types': node_types,
                'type_distribution': {
                    t: list(node_types.values()).count(t) / len(node_types)
                    for t in set(node_types.values())
                }
            }
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
                
            print(f"Successfully exported cell types to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting cell types: {e}")
            return False