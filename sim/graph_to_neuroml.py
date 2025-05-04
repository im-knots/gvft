import networkx as nx
import numpy as np
import os
import json
from collections import Counter
import xml.etree.ElementTree as ET
import datetime
from lxml import etree

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
    
    def create_neuroml_document(self, network_id="EvolvedNetwork"):
        """Create a basic NeuroML2 document with standard headers."""
        # Create the root element with appropriate namespaces
        neuroml = etree.Element(
            "neuroml",
            nsmap={
                None: "http://www.neuroml.org/schema/neuroml2",
                "xsi": "http://www.w3.org/2001/XMLSchema-instance",
            },
            attrib={
                "{http://www.w3.org/2001/XMLSchema-instance}schemaLocation": 
                    "http://www.neuroml.org/schema/neuroml2 https://raw.githubusercontent.com/NeuroML/NeuroML2/development/Schemas/NeuroML2/NeuroML_v2beta4.xsd",
                "id": network_id
            }
        )
        
        # Add creation timestamp and notes
        notes = etree.SubElement(neuroml, "notes")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        notes.text = f"NeuroML2 network generated from GVFT simulation on {timestamp}\n"
        
        return neuroml
    
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
            
            # Assign synapse properties
            synapse_props = self.assign_synapse_properties(G, node_types)
            
            # Create the NeuroML document
            network_id = os.path.splitext(os.path.basename(output_path))[0]
            neuroml_doc = self.create_neuroml_document(network_id)
            
            # Create cell types
            for cell_type, params in self.default_params.items():
                cell_id = f"{cell_type}_cell"
                
                # Create izhikevich cell
                cell = etree.SubElement(
                    neuroml_doc, 
                    "izhikevichCell", 
                    id=cell_id,
                    v0=f"{params['v_rest']}mV",
                    thresh=f"{params['threshold']}mV",
                    a="0.02",
                    b="0.2",
                    c=f"{params['v_rest']}mV",
                    d="2.0",
                    peak="30.0mV"
                )
            
            # Create synapse types
            etree.SubElement(
                neuroml_doc,
                "expTwoSynapse",
                id="exc_syn",
                gbase="5nS",
                tauDecay="2ms",
                tauRise="0.5ms",
                erev="0mV"
            )
            
            etree.SubElement(
                neuroml_doc,
                "expTwoSynapse",
                id="inh_syn",
                gbase="5nS",
                tauDecay="5ms",
                tauRise="1ms",
                erev="-80mV"
            )
            
            # Create network container
            network = etree.SubElement(
                neuroml_doc,
                "network",
                id=f"{network_id}_net",
                type="networkWithTemperature",
                temperature="36.0 degC"
            )
            
            # Group nodes by cell type
            populations = {}
            for cell_type in self.cell_types:
                populations[cell_type] = [n for n, t in node_types.items() if t == cell_type]
            
            # Create populations
            pop_info = {}
            for cell_type, nodes in populations.items():
                if not nodes:
                    continue
                    
                pop_id = f"{cell_type}_population"
                
                # Create population
                pop = etree.SubElement(
                    network,
                    "population",
                    id=pop_id,
                    component=f"{cell_type}_cell",
                    type="populationList",
                    size=str(len(nodes))
                )
                
                # Add instances
                for i, node_id in enumerate(nodes):
                    instance = etree.SubElement(pop, "instance", id=str(i))
                    
                    # Get position if available
                    pos = None
                    if 'pos' in G.nodes[node_id]:
                        pos = G.nodes[node_id]['pos']
                    else:
                        # Default position
                        pos = (float(node_id) / len(G.nodes) * 200 - 100, 0)
                    
                    # Add location
                    etree.SubElement(
                        instance,
                        "location",
                        x=f"{pos[0]:.6f}",
                        y=f"{pos[1]:.6f}",
                        z="0"
                    )
                
                # Store mapping of node IDs to population indices
                pop_info[cell_type] = {
                    'id': pop_id,
                    'nodes': nodes
                }
            
            # Create a lookup for node to population mapping
            node_to_pop = {}
            for cell_type, info in pop_info.items():
                for i, node_id in enumerate(info['nodes']):
                    node_to_pop[node_id] = {
                        'pop_id': info['id'],
                        'index': i
                    }
            
            # Group connections by source/target population pairs
            proj_connections = {}
            
            for conn_id, props in synapse_props.items():
                source = props['source']
                target = props['target']
                
                # Skip if source or target not in mapping
                if source not in node_to_pop or target not in node_to_pop:
                    continue
                
                source_pop = node_to_pop[source]['pop_id']
                source_idx = node_to_pop[source]['index']
                
                target_pop = node_to_pop[target]['pop_id']
                target_idx = node_to_pop[target]['index']
                
                # Create projection key
                proj_key = f"{source_pop}_{target_pop}_{'exc' if props['is_excitatory'] else 'inh'}"
                
                # Add to projection group
                if proj_key not in proj_connections:
                    proj_connections[proj_key] = []
                
                proj_connections[proj_key].append({
                    'id': conn_id,
                    'pre_cell': source_idx,
                    'post_cell': target_idx,
                    'weight': props['weight'],
                    'delay': props['delay']
                })
            
            # Create projections
            for proj_key, connections in proj_connections.items():
                if not connections:
                    continue
                
                # Parse projection key
                parts = proj_key.split('_')
                source_pop = parts[0]
                target_pop = parts[1]
                syn_type = "exc_syn" if parts[2] == "exc" else "inh_syn"
                
                # Create projection
                proj = etree.SubElement(
                    network,
                    "projection",
                    id=proj_key,
                    presynapticPopulation=source_pop,
                    postsynapticPopulation=target_pop,
                    synapse=syn_type
                )
                
                # Add connections
                for conn in connections:
                    etree.SubElement(
                        proj,
                        "connection",
                        id=conn['id'],
                        preCellId=f"../{source_pop}/{conn['pre_cell']}/{source_pop}",
                        postCellId=f"../{target_pop}/{conn['post_cell']}/{target_pop}",
                        weight=str(conn['weight']),
                        delay=f"{conn['delay']}ms"
                    )
            
            # Write to file
            tree = etree.ElementTree(neuroml_doc)
            tree.write(output_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
            
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