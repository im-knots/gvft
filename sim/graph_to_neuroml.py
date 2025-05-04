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
    
    This class is specifically designed to work with C. elegans pharyngeal
    nervous system conventions, matching the format in PharyngealNetwork.net.nml.
    """
    
    def __init__(self, original_network=None):
        """
        Initialize the converter with an optional original network for reference.
        
        Args:
            original_network (nx.Graph, optional): Original reference network
        """
        self.original_network = original_network
        
        # Pharyngeal neuron classes with their prefixes
        # Based on C. elegans pharyngeal nervous system
        self.cell_classes = {
            'M': ['M1', 'M2L', 'M2R', 'M3L', 'M3R', 'M4', 'M5'],  # Motor neurons
            'I': ['I1L', 'I1R', 'I2L', 'I2R', 'I3', 'I4', 'I5', 'I6'],  # Sensory neurons
            'MC': ['MCL', 'MCR'],  # Interneurons (marginal cells)
            'MI': ['MI'],  # Interneurons (motor-interneurons)
            'NSM': ['NSML', 'NSMR']  # Neuromodulatory/Serotonergic
        }
        
        # Flatten all possible neuron types
        self.all_neuron_types = []
        for types in self.cell_classes.values():
            self.all_neuron_types.extend(types)
        
        # Map neuron prefixes to cell types for classification
        self.prefix_to_type = {
            'M': 'motor',
            'I': 'sensory',
            'MC': 'inter',
            'MI': 'inter',
            'NSM': 'inter'
        }
        
        # Target distribution based on C. elegans pharyngeal nervous system
        # Calculated from the actual counts in the network
        self.target_distribution = {
            'motor': 0.35,    # 7/20 = 35% motor neurons
            'sensory': 0.4,   # 8/20 = 40% sensory neurons  
            'inter': 0.25     # 5/20 = 25% interneurons (MC + MI + NSM)
        }
        
        # Load cell parameters from NeuroML file or use defaults
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
        
        # Synapse types from the original NeuroML file
        self.synapse_types = {
            'excitatory': 'Acetylcholine',
            'inhibitory': 'Glutamate',
            'modulatory': 'Serotonin_Glutamate',
            'gap_junction': 'Generic_GJ'
        }
        
        # Somatic neurons that connect to pharyngeal nervous system
        # Based on the original PharyngealNetwork.net.nml file
        self.somatic_connections = [
            # Format: (pharyngeal_neuron, somatic_neuron, direction, synapse_type)
            ("I1L", "RMDDL", "outgoing", "Acetylcholine"),
            ("I1L", "RMDVL", "outgoing", "Acetylcholine"),
            ("I1R", "RMDDR", "outgoing", "Acetylcholine"),
            ("I1R", "RMDVR", "outgoing", "Acetylcholine"),
            ("I2L", "URXL", "outgoing", "Glutamate"),
            ("I2R", "URXR", "outgoing", "Glutamate"),
            ("I1L", "AVKL", "incoming", "Acetylcholine"),
            ("I1R", "AVKR", "incoming", "Acetylcholine"),
            ("I1L", "RIS", "incoming", "Glutamate"),
            ("I1R", "RIS", "incoming", "Glutamate"),
            ("I2L", "AVAL", "incoming", "Acetylcholine"),
            ("I2R", "AVAR", "incoming", "Acetylcholine"),
            ("I4", "SMBVL", "incoming", "Acetylcholine"),
            ("I4", "SMBVR", "incoming", "Acetylcholine")
        ]
        
        # Collect unique somatic neurons for includes
        self.somatic_neurons = sorted(list(set([conn[1] for conn in self.somatic_connections])))
    
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
            dict: Dictionary mapping node IDs to cell types and class names
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
        basic_assignments = {}
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
                basic_assignments[node] = cell_type
                remaining_nodes.remove(node)
        
        # Assign any remaining nodes to their highest-scoring type
        for node in remaining_nodes:
            node_scores = scores[node]
            basic_assignments[node] = max(node_scores, key=node_scores.get)
        
        # Now map from basic types to the specific pharyngeal neuron classes
        detailed_assignments = {}
        
        # Count how many neurons we've already assigned of each detailed type
        detailed_counts = {neuron_type: 0 for neuron_type in self.all_neuron_types}
        
        # Assign detailed cell types
        for node, basic_type in basic_assignments.items():
            # Get all possible detailed cell types for this basic type
            possible_classes = []
            for prefix, types in self.cell_classes.items():
                if self.prefix_to_type.get(prefix) == basic_type:
                    possible_classes.extend(types)
            
            # If we have no detailed classes for this type, just use a default
            if not possible_classes:
                detailed_assignments[node] = {
                    'basic_type': basic_type,
                    'cell_class': basic_type.upper() + str(detailed_counts.get(basic_type, 0) + 1)
                }
                continue
            
            # Pick the least used neuron type to ensure even distribution
            counts = [(t, detailed_counts[t]) for t in possible_classes]
            counts.sort(key=lambda x: x[1])  # Sort by count (ascending)
            
            chosen_type = counts[0][0]
            detailed_counts[chosen_type] += 1
            
            # Store both basic and detailed assignments
            detailed_assignments[node] = {
                'basic_type': basic_type,
                'cell_class': chosen_type
            }
        
        return detailed_assignments
    
    def validate_pharyngeal_interface(self, node_types):
        """
        Validate that all required interface neurons are present in the network
        
        Args:
            node_types (dict): Dictionary mapping node IDs to cell types
            
        Returns:
            list: List of missing required neurons
        """
        # Essential interface neurons that connect to somatic nervous system
        required_neurons = [pharyngeal_neuron for pharyngeal_neuron, _, _, _ in self.somatic_connections]
        # Remove duplicates
        required_neurons = list(set(required_neurons))
        
        # Check if all required neurons are present
        present_neurons = set()
        for node_id, type_info in node_types.items():
            cell_class = type_info['cell_class']
            present_neurons.add(cell_class)
        
        # Find missing neurons
        missing_neurons = []
        for neuron in required_neurons:
            if neuron not in present_neurons:
                missing_neurons.append(neuron)
        
        return missing_neurons
    
    def assign_synapse_properties(self, G, node_types):
        """
        Assign synaptic properties to the connections in the graph
        
        Args:
            G (nx.Graph): The network graph
            node_types (dict): Dictionary mapping node IDs to cell type information
            
        Returns:
            dict: Dictionary of dictionaries with edge properties
        """
        edge_props = {}
        
        # For each edge in the graph
        for source, target, data in G.edges(data=True):
            # Get cell types of source and target
            source_type = node_types.get(source, {}).get('basic_type', 'inter')
            target_type = node_types.get(target, {}).get('basic_type', 'inter')
            
            # Get source and target class
            source_class = node_types.get(source, {}).get('cell_class', '')
            target_class = node_types.get(target, {}).get('cell_class', '') 
            
            # Get edge weight or default to 1.0
            weight = data.get('weight', 1.0)
            
            # Determine if connection is excitatory, inhibitory, or modulatory
            connection_type = 'excitatory'  # Default
            
            # Apply consistent rules based on source type
            if source_class.startswith('NSM'):
                # NSM neurons are serotonergic/modulatory
                connection_type = 'modulatory'
            elif source_type == 'motor':
                if target_type == 'motor':
                    # Motor to motor has a chance of inhibitory
                    connection_type = 'inhibitory' if np.random.random() < 0.4 else 'excitatory'
                else:
                    connection_type = 'excitatory'
            elif source_type == 'sensory':
                if target_type == 'inter':
                    # Sensory to inter is typically excitatory
                    connection_type = 'excitatory'
                else:
                    # Sensory to others can be either
                    connection_type = np.random.choice(['excitatory', 'inhibitory'], 
                                                    p=[0.7, 0.3])
            elif source_type == 'inter':
                # Interneurons make both types, with a bias
                if source_class.startswith('MC'):
                    connection_type = 'excitatory'  # MC cells are excitatory
                else:
                    connection_type = np.random.choice(['excitatory', 'inhibitory', 'modulatory'], 
                                                    p=[0.5, 0.3, 0.2])
            
            # Determine synapse type based on connection type
            synapse_type = self.synapse_types.get(connection_type, 'Acetylcholine')
            
            # Decide if this is a gap junction (electrical synapse)
            # In pharyngeal system, these are often between same-type cells or symmetrical pairs
            is_gap_junction = False
            if (source_type == target_type or 
                (source_class.endswith('L') and target_class.endswith('R') and 
                source_class[:-1] == target_class[:-1]) or
                (source_class.endswith('R') and target_class.endswith('L') and 
                source_class[:-1] == target_class[:-1])):
                # Symmetrically positioned cells often have gap junctions
                # With a certain probability
                is_gap_junction = np.random.random() < 0.3
            
            # If it's a gap junction, override synapse type
            if is_gap_junction:
                synapse_type = self.synapse_types['gap_junction']
            
            # Calculate delay based on connection weight
            # lower weights get higher delays
            delay = 0.5 + (1.0 - min(weight, 1.0)) * 1.5  # 0.5 to 2.0 ms
            
            # Assign a unique ID to the connection
            conn_id = f"{source}_{target}"
            
            edge_props[conn_id] = {
                'source': source,
                'target': target,
                'weight': weight,
                'delay': delay,
                'synapse_type': synapse_type,
                'is_gap_junction': is_gap_junction
            }
            
        return edge_props
    
    def create_neuroml_document(self, network_id="EvolvedNetwork"):
        """Create a basic NeuroML2 document with standard headers."""
        # Sanitize the network_id to conform to NmlId pattern [a-zA-Z_][a-zA-Z0-9_]*
        sanitized_id = network_id
        # Remove file extension if present
        if sanitized_id.endswith('.net.nml'):
            sanitized_id = sanitized_id[:-8]
        elif sanitized_id.endswith('.nml'):
            sanitized_id = sanitized_id[:-4]
        
        # Replace invalid characters with underscores
        sanitized_id = ''.join(c if c.isalnum() or c == '_' else '_' for c in sanitized_id)
        
        # Ensure it starts with a letter or underscore
        if sanitized_id and not (sanitized_id[0].isalpha() or sanitized_id[0] == '_'):
            sanitized_id = 'gvft_' + sanitized_id
        
        # If empty after sanitization, use a default
        if not sanitized_id:
            sanitized_id = "gvft_network"
        
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
                "id": sanitized_id
            }
        )
        
        # Add creation timestamp and notes
        notes = etree.SubElement(neuroml, "notes")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        notes.text = f"NeuroML2 network generated from GVFT simulation on {timestamp}\n"
        
        # Include references to the somatic neurons needed for interface
        for neuron in self.somatic_neurons:
            # Note that these includes assume the cell definitions exist in the same directory
            etree.SubElement(neuroml, "include", href=f"{neuron}.cell.nml")
        
        return neuroml
    
    def export_to_neuroml(self, G, output_path, node_types=None):
        """
        Export a graph to a NeuroML file that matches the C. elegans pharyngeal
        nervous system format, including connections to the somatic nervous system
        
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
            
            # Validate pharyngeal interface neurons
            missing_neurons = self.validate_pharyngeal_interface(node_types)
            if missing_neurons:
                print(f"Warning: Missing required interface neurons: {missing_neurons}")
                print("Some connections to somatic nervous system may be incomplete.")
            
            # Assign synapse properties
            synapse_props = self.assign_synapse_properties(G, node_types)
            
            # Create the NeuroML document
            network_id = os.path.splitext(os.path.basename(output_path))[0]
            neuroml_doc = self.create_neuroml_document(network_id)
            
            # Include references to necessary cell and synapse files
            # Note: These files should exist alongside the NeuroML file
            for cell_class in set(info['cell_class'] for info in node_types.values()):
                etree.SubElement(neuroml_doc, "include", href=f"{cell_class}.cell.nml")
            
            # Include synapse references
            for synapse_type in set(self.synapse_types.values()):
                etree.SubElement(neuroml_doc, "include", href=f"{synapse_type}.synapse.nml")
            
            # Create network container
            # Sanitize the ID first
            sanitized_id = network_id
            if sanitized_id.endswith('.net.nml'):
                sanitized_id = sanitized_id[:-8]
            elif sanitized_id.endswith('.nml'):
                sanitized_id = sanitized_id[:-4]
            sanitized_id = ''.join(c if c.isalnum() or c == '_' else '_' for c in sanitized_id)
            if sanitized_id and not (sanitized_id[0].isalpha() or sanitized_id[0] == '_'):
                sanitized_id = 'gvft_' + sanitized_id
            if not sanitized_id:
                sanitized_id = "gvft_network"
                
            network = etree.SubElement(
                neuroml_doc,
                "network",
                id=sanitized_id,
                type="networkWithTemperature",
                temperature="20.0 degC"
            )
            
            # Group nodes by cell_class
            population_nodes = {}
            for node_id, type_info in node_types.items():
                cell_class = type_info['cell_class']
                if cell_class not in population_nodes:
                    population_nodes[cell_class] = []
                population_nodes[cell_class].append(node_id)
            
            # Create populations - one for each cell class
            for cell_class, node_ids in population_nodes.items():
                # Create population
                pop = etree.SubElement(
                    network,
                    "population",
                    id=cell_class,
                    component=cell_class,
                    type="populationList",
                    size=str(len(node_ids))
                )
                
                # Add color annotation
                annotation = etree.SubElement(pop, "annotation")
                # Generate deterministic color based on cell class
                color_hash = sum(ord(c) for c in cell_class) % 100
                r = (color_hash * 13) % 100 / 100
                g = (color_hash * 17) % 100 / 100
                b = (color_hash * 19) % 100 / 100
                etree.SubElement(annotation, "property", tag="color", value=f"{r} {g} {b}")
                
                # Add instances
                for i, node_id in enumerate(node_ids):
                    instance = etree.SubElement(pop, "instance", id=str(i))
                    
                    # Get position if available
                    pos = None
                    if 'pos' in G.nodes[node_id]:
                        pos = G.nodes[node_id]['pos']
                    else:
                        # Default position in the plane
                        pos = (0.0, 0.0, 0.0)
                    
                    # Add location
                    etree.SubElement(
                        instance,
                        "location",
                        x=str(pos[0]),
                        y=str(pos[1]),
                        z=str(pos[2] if len(pos) > 2 else 0.0)
                    )
            
            # Create somatic neuron populations if they're not already in the network
            for somatic_neuron in self.somatic_neurons:
                if somatic_neuron not in population_nodes:
                    # Create population for this somatic neuron
                    pop = etree.SubElement(
                        network,
                        "population",
                        id=somatic_neuron,
                        component=somatic_neuron,
                        type="populationList",
                        size="1"
                    )
                    
                    # Add color annotation (predictable but different for somatic)
                    annotation = etree.SubElement(pop, "annotation")
                    color_hash = sum(ord(c) for c in somatic_neuron) % 100
                    r = (color_hash * 11) % 100 / 100
                    g = (color_hash * 23) % 100 / 100
                    b = (color_hash * 7) % 100 / 100
                    etree.SubElement(annotation, "property", tag="color", value=f"{r} {g} {b}")
                    
                    # Add instance with a default position outside pharyngeal network
                    instance = etree.SubElement(pop, "instance", id="0")
                    etree.SubElement(
                        instance,
                        "location",
                        x="5.0",
                        y="0.0",
                        z="0.0"
                    )
            
            # --- IMPORTANT CHANGE: Reorder projections to follow NeuroML schema ---
            
            # First collect all projections by type to ensure proper ordering
            electrical_projections = []  # For electricalProjection elements
            chemical_projections = []  # For projection elements
            
            # Process internal connections and separate by type
            for conn_id, props in synapse_props.items():
                source_node = props['source']
                target_node = props['target']
                
                # Get source and target cell classes
                source_class = node_types[source_node]['cell_class']
                target_class = node_types[target_node]['cell_class']
                
                # Get indices within their respective populations
                source_idx = population_nodes[source_class].index(source_node)
                target_idx = population_nodes[target_class].index(target_node)
                
                synapse_type = props['synapse_type']
                is_gap_junction = props['is_gap_junction']
                
                if is_gap_junction:
                    electrical_projections.append({
                        'source_class': source_class,
                        'target_class': target_class,
                        'source_idx': source_idx,
                        'target_idx': target_idx,
                        'weight': props['weight'],
                        'delay': props['delay']
                    })
                else:
                    chemical_projections.append({
                        'source_class': source_class,
                        'target_class': target_class,
                        'source_idx': source_idx,
                        'target_idx': target_idx,
                        'weight': props['weight'],
                        'delay': props['delay'],
                        'synapse_type': synapse_type
                    })
            
            # Process somatic interface connections (all chemical)
            for pharyngeal_neuron, somatic_neuron, direction, synapse_type in self.somatic_connections:
                # Check if the pharyngeal neuron exists
                pharyngeal_found = False
                for cell_class in population_nodes.keys():
                    if cell_class == pharyngeal_neuron:
                        pharyngeal_found = True
                        break
                
                if not pharyngeal_found:
                    continue
                    
                # Add to chemical_projections with appropriate source/target
                if direction == "outgoing":  # pharyngeal -> somatic
                    chemical_projections.append({
                        'source_class': pharyngeal_neuron,
                        'target_class': somatic_neuron,
                        'source_idx': 0,
                        'target_idx': 0,
                        'synapse_type': synapse_type,
                        'is_interface': True
                    })
                else:  # incoming: somatic -> pharyngeal
                    chemical_projections.append({
                        'source_class': somatic_neuron,
                        'target_class': pharyngeal_neuron,
                        'source_idx': 0,
                        'target_idx': 0,
                        'synapse_type': synapse_type,
                        'is_interface': True
                    })
            projection_counter = 0      
            # 1. Add chemical projections
            chemical_proj_map = {}  # To track existing projections
            
            for proj_data in chemical_projections:
                source_class = proj_data['source_class']
                target_class = proj_data['target_class']
                source_idx = proj_data['source_idx']
                target_idx = proj_data['target_idx']
                synapse_type = proj_data['synapse_type']
                is_interface = proj_data.get('is_interface', False)
                
                # Create different projection IDs for interface vs internal
                if is_interface:
                    proj_id = f"NC_{source_class}_{target_class}"
                else:
                    proj_id = f"NCXLS_{source_class}_{target_class}"
                
                # Check if we've already created this projection
                if proj_id not in chemical_proj_map:
                    # Create new projection
                    proj = etree.SubElement(
                        network,
                        "projection",
                        id=proj_id,
                        presynapticPopulation=source_class,
                        postsynapticPopulation=target_class,
                        synapse=synapse_type
                    )
                    chemical_proj_map[proj_id] = proj
                else:
                    proj = chemical_proj_map[proj_id]
                
                # Add connection
                conn = etree.SubElement(
                    proj,
                    "connection",
                    id=str(projection_counter),
                    preCellId=f"../{source_class}/{source_idx}/{source_class}",
                    postCellId=f"../{target_class}/{target_idx}/{target_class}"
                )
                
                # Add synaptic location parameters
                pre_segment = np.random.randint(1, 10)
                post_segment = np.random.randint(1, 10)
                pre_frac = np.random.random()
                post_frac = np.random.random()
                
                conn.attrib["preSegmentId"] = str(pre_segment)
                conn.attrib["postSegmentId"] = str(post_segment)
                conn.attrib["preFractionAlong"] = str(pre_frac)
                conn.attrib["postFractionAlong"] = str(post_frac)
                
                projection_counter += 1

            # Now add electrical projections
            # 2. Add electrical projections (gap junctions)
            electrical_proj_map = {}  # To track existing projections
            
            for proj_data in electrical_projections:
                source_class = proj_data['source_class']
                target_class = proj_data['target_class']
                source_idx = proj_data['source_idx']
                target_idx = proj_data['target_idx']
                
                # Create projection ID
                proj_id = f"NCXLS_{source_class}_{target_class}_GJ"
                
                # Check if we've already created this projection
                if proj_id not in electrical_proj_map:
                    # Create new electrical projection
                    proj = etree.SubElement(
                        network,
                        "electricalProjection",
                        id=proj_id,
                        presynapticPopulation=source_class,
                        postsynapticPopulation=target_class
                    )
                    electrical_proj_map[proj_id] = proj
                else:
                    proj = electrical_proj_map[proj_id]
                
                # Add connection
                conn = etree.SubElement(
                    proj,
                    "electricalConnection",
                    id=str(projection_counter),
                    preCell=str(source_idx),
                    postCell=str(target_idx),
                    synapse="Generic_GJ"
                )
                
                # Add synaptic location parameters
                pre_segment = np.random.randint(1, 10)
                post_segment = np.random.randint(1, 10)
                pre_frac = np.random.random()
                post_frac = np.random.random()
                
                conn.attrib["preSegment"] = str(pre_segment)
                conn.attrib["postSegment"] = str(post_segment)
                conn.attrib["preFractionAlong"] = str(pre_frac)
                conn.attrib["postFractionAlong"] = str(post_frac)
                
                projection_counter += 1
            
            # Write to file
            tree = etree.ElementTree(neuroml_doc)
            tree.write(output_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
            
            print(f"Successfully exported to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting to NeuroML: {e}")
            return False
    
    def _create_internal_projections(self, network, synapse_props, node_types, population_nodes):
        """Create internal projections within the pharyngeal nervous system"""
        projection_counter = 0
        for conn_id, props in synapse_props.items():
            source_node = props['source']
            target_node = props['target']
            
            # Get source and target cell classes
            source_class = node_types[source_node]['cell_class']
            target_class = node_types[target_node]['cell_class']
            
            # Get indices within their respective populations
            source_idx = population_nodes[source_class].index(source_node)
            target_idx = population_nodes[target_class].index(target_node)
            
            # Creating projection ID following the conventions in PharyngealNetwork.net.nml
            # Format: NCXLS_<SOURCE>_<TARGET> for chemical synapses
            # Format: NCXLS_<SOURCE>_<TARGET>_GJ for gap junctions
            synapse_type = props['synapse_type']
            is_gap_junction = props['is_gap_junction']
            
            if is_gap_junction:
                proj_id = f"NCXLS_{source_class}_{target_class}_GJ"
                
                # Check if projection already exists
                existing_proj = network.find(f".//electricalProjection[@id='{proj_id}']")
                if existing_proj is None:
                    # Create new electrical projection
                    proj = etree.SubElement(
                        network,
                        "electricalProjection",
                        id=proj_id,
                        presynapticPopulation=source_class,
                        postsynapticPopulation=target_class
                    )
                else:
                    proj = existing_proj
                
                # Add connection
                conn = etree.SubElement(
                    proj,
                    "electricalConnection",
                    id=str(projection_counter),
                    preCell=str(source_idx),
                    postCell=str(target_idx),
                    synapse="Generic_GJ"
                )
                
                # Add synaptic location parameters
                pre_segment = np.random.randint(1, 10)
                post_segment = np.random.randint(1, 10)
                pre_frac = np.random.random()
                post_frac = np.random.random()
                
                conn.attrib["preSegment"] = str(pre_segment)
                conn.attrib["postSegment"] = str(post_segment)
                conn.attrib["preFractionAlong"] = str(pre_frac)
                conn.attrib["postFractionAlong"] = str(post_frac)
                
            else:
                # Chemical synapse
                proj_id = f"NCXLS_{source_class}_{target_class}"
                
                # Check if projection already exists
                existing_proj = network.find(f".//projection[@id='{proj_id}']")
                if existing_proj is None:
                    # Create new projection
                    proj = etree.SubElement(
                        network,
                        "projection",
                        id=proj_id,
                        presynapticPopulation=source_class,
                        postsynapticPopulation=target_class,
                        synapse=synapse_type
                    )
                else:
                    proj = existing_proj
                
                # Add connection
                conn = etree.SubElement(
                    proj,
                    "connection",
                    id=str(projection_counter),
                    preCellId=f"../{source_class}/{source_idx}/{source_class}",
                    postCellId=f"../{target_class}/{target_idx}/{target_class}"
                )
                
                # Add synaptic location parameters
                pre_segment = np.random.randint(1, 10)
                post_segment = np.random.randint(1, 10)
                pre_frac = np.random.random()
                post_frac = np.random.random()
                
                conn.attrib["preSegmentId"] = str(pre_segment)
                conn.attrib["postSegmentId"] = str(post_segment)
                conn.attrib["preFractionAlong"] = str(pre_frac)
                conn.attrib["postFractionAlong"] = str(post_frac)
            
            projection_counter += 1
    
    def _create_interface_connections(self, network, node_types, population_nodes):
        """Create connections between pharyngeal and somatic nervous system"""
        # Counter for connection IDs
        projection_counter = 1000  # Start from 1000 to avoid conflicts
        
        # Create the interface connections
        for pharyngeal_neuron, somatic_neuron, direction, synapse_type in self.somatic_connections:
            # Check if the pharyngeal neuron exists in our node types
            pharyngeal_found = False
            for cell_class in population_nodes.keys():
                if cell_class == pharyngeal_neuron:
                    pharyngeal_found = True
                    break
            
            if not pharyngeal_found:
                print(f"Warning: Pharyngeal neuron {pharyngeal_neuron} not found in the network, skipping interface connection.")
                continue
                
            # For outgoing connections (pharyngeal -> somatic)
            if direction == "outgoing":
                # Create projection ID
                proj_id = f"NC_{pharyngeal_neuron}_{somatic_neuron}"
                
                # Check if projection already exists
                existing_proj = network.find(f".//projection[@id='{proj_id}']")
                if existing_proj is None:
                    # Create new projection
                    proj = etree.SubElement(
                        network,
                        "projection",
                        id=proj_id,
                        presynapticPopulation=pharyngeal_neuron,
                        postsynapticPopulation=somatic_neuron,
                        synapse=synapse_type
                    )
                else:
                    proj = existing_proj
                
                # Add connection - note we use the first neuron in each population
                conn = etree.SubElement(
                    proj,
                    "connection",
                    id=str(projection_counter),
                    preCellId=f"../{pharyngeal_neuron}/0/{pharyngeal_neuron}",
                    postCellId=f"../{somatic_neuron}/0/{somatic_neuron}"
                )
                
                # Add synaptic location parameters
                pre_segment = np.random.randint(1, 10)
                post_segment = np.random.randint(1, 10)
                pre_frac = np.random.random()
                post_frac = np.random.random()
                
                conn.attrib["preSegmentId"] = str(pre_segment)
                conn.attrib["postSegmentId"] = str(post_segment)
                conn.attrib["preFractionAlong"] = str(pre_frac)
                conn.attrib["postFractionAlong"] = str(post_frac)
                
                projection_counter += 1
                
            # For incoming connections (somatic -> pharyngeal)
            elif direction == "incoming":
                # Create projection ID
                proj_id = f"NC_{somatic_neuron}_{pharyngeal_neuron}"
                
                # Check if projection already exists
                existing_proj = network.find(f".//projection[@id='{proj_id}']")
                if existing_proj is None:
                    # Create new projection
                    proj = etree.SubElement(
                        network,
                        "projection",
                        id=proj_id,
                        presynapticPopulation=somatic_neuron,
                        postsynapticPopulation=pharyngeal_neuron,
                        synapse=synapse_type
                    )
                else:
                    proj = existing_proj
                
                # Add connection
                conn = etree.SubElement(
                    proj,
                    "connection",
                    id=str(projection_counter),
                    preCellId=f"../{somatic_neuron}/0/{somatic_neuron}",
                    postCellId=f"../{pharyngeal_neuron}/0/{pharyngeal_neuron}"
                )
                
                # Add synaptic location parameters
                pre_segment = np.random.randint(1, 10)
                post_segment = np.random.randint(1, 10)
                pre_frac = np.random.random()
                post_frac = np.random.random()
                
                conn.attrib["preSegmentId"] = str(pre_segment)
                conn.attrib["postSegmentId"] = str(post_segment)
                conn.attrib["preFractionAlong"] = str(pre_frac)
                conn.attrib["postFractionAlong"] = str(post_frac)
                
                projection_counter += 1