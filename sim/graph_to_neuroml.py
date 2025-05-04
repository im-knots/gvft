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
        
        # Synapse types for the pharyngeal nervous system
        # Exact names from the organicNeuroML2 directory
        self.synapse_types = {
            'excitatory': 'Acetylcholine',
            'inhibitory': 'Glutamate',
            'modulatory': 'Serotonin_Glutamate',
            'gap_junction': 'Generic_GJ'
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
            # Convert to undirected for community detection
            G_undirected = G.to_undirected()
            # Attempt to find communities with Louvain method
            communities = list(nx.community.louvain_communities(G_undirected, resolution=1.0))
            
            # Convert to dictionary format
            community_map = {}
            for idx, comm in enumerate(communities):
                for node in comm:
                    community_map[node] = idx
            communities = community_map
        except Exception as e:
            # Fall back to connected components if community detection fails
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
        
        # Group nodes by community to assign similar cell types to nodes in the same community
        community_nodes = {}
        for node, m in metrics.items():
            community = m['community']
            if community not in community_nodes:
                community_nodes[community] = []
            community_nodes[community].append(node)
        
        # Assign detailed cell types by community
        for community, nodes in community_nodes.items():
            # Get basic cell types for nodes in this community
            community_types = [basic_assignments[node] for node in nodes]
            most_common_type = Counter(community_types).most_common(1)[0][0]
            
            # Get all possible detailed cell types for this basic type
            possible_classes = []
            for prefix, types in self.cell_classes.items():
                if self.prefix_to_type.get(prefix) == most_common_type:
                    possible_classes.extend(types)
            
            # If we have no detailed classes for this type, use a different type
            if not possible_classes:
                for prefix, types in self.cell_classes.items():
                    possible_classes.extend(types)
            
            # Sort by current usage (prefer least used types)
            possible_classes.sort(key=lambda t: detailed_counts.get(t, 0))
            
            # Assign cell types to nodes in this community
            for node in nodes:
                basic_type = basic_assignments[node]
                
                # Find a matching class that hasn't been used too much
                chosen_type = None
                for t in possible_classes:
                    prefix = t[0] if not t.startswith(('MC', 'NSM')) else t[:3]
                    if self.prefix_to_type.get(prefix) == basic_type:
                        chosen_type = t
                        break
                
                # If no matching class, use the first one that matches the basic type
                if chosen_type is None:
                    matching_types = []
                    for t in self.all_neuron_types:
                        prefix = t[0] if not t.startswith(('MC', 'NSM')) else t[:3]
                        if self.prefix_to_type.get(prefix) == basic_type:
                            matching_types.append(t)
                    
                    if matching_types:
                        matching_types.sort(key=lambda t: detailed_counts.get(t, 0))
                        chosen_type = matching_types[0]
                    else:
                        # Last resort: use any available type
                        available_types = sorted(self.all_neuron_types, 
                                               key=lambda t: detailed_counts.get(t, 0))
                        chosen_type = available_types[0] if available_types else "M1"
                
                # Update count and store assignment
                detailed_counts[chosen_type] = detailed_counts.get(chosen_type, 0) + 1
                detailed_assignments[node] = {
                    'basic_type': basic_type,
                    'cell_class': chosen_type
                }
        
        return detailed_assignments
    
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
        
        # Count the different connection types for reporting
        connection_counts = {
            'excitatory': 0,
            'inhibitory': 0,
            'modulatory': 0,
            'gap_junction': 0
        }
        
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
            # IMPORTANT: Increase probability of gap junctions
            is_gap_junction = False
            if (source_type == target_type or 
                (source_class.endswith('L') and target_class.endswith('R') and 
                source_class[:-1] == target_class[:-1]) or
                (source_class.endswith('R') and target_class.endswith('L') and 
                source_class[:-1] == target_class[:-1])):
                # Symmetrically positioned cells often have gap junctions
                # Increase probability from 0.3 to 0.5
                is_gap_junction = np.random.random() < 0.5
            
            # Additional rule: Some specific cell classes are known to form gap junctions
            if ('M' in source_class and 'M' in target_class) or ('MC' in source_class and 'MC' in target_class):
                # Motor neurons and marginal cells often form gap junctions with their class
                is_gap_junction = np.random.random() < 0.6
            
            # If it's a gap junction, override synapse type
            if is_gap_junction:
                synapse_type = self.synapse_types['gap_junction']
                connection_counts['gap_junction'] += 1
            else:
                connection_counts[connection_type] += 1
            
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
        
        # Print statistics
        print(f"Connection type distribution:")
        print(f"  Excitatory: {connection_counts['excitatory']}")
        print(f"  Inhibitory: {connection_counts['inhibitory']}")
        print(f"  Modulatory: {connection_counts['modulatory']}")
        print(f"  Gap Junctions: {connection_counts['gap_junction']}")
        
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
    
    def get_synapse_filename(self, synapse_type):
        """
        Get the correct filename for a synapse type based on the organicNeuroML2 directory
        
        Args:
            synapse_type (str): The synapse type name
            
        Returns:
            str: Synapse filename with extension
        """
        # Map to exact filenames as they appear in the organicNeuroML2 directory
        synapse_files = {
            'Acetylcholine': 'Acetylcholine.synapse.nml',
            'Glutamate': 'Glutamate.synapse.nml',
            'Serotonin': 'Serotonin.synapse.nml',
            'Serotonin_Glutamate': 'Serotonin_Glutamate.synapse.nml', 
            'Serotonin_Acetylcholine': 'Serotonin_Acetylcholine.synapse.nml',
            'Acetylcholine_Tyramine': 'Acetylcholine_Tyramine.synapse.nml',
            'Dopamine': 'Dopamine.synapse.nml',
            'GABA': 'GABA.synapse.nml',
            'FMRFamide': 'FMRFamide.synapse.nml',
            'Generic_GJ': 'Generic_GJ.nml',
            'Octapamine': 'Octapamine.synapse.nml'
        }
        
        return synapse_files.get(synapse_type, f"{synapse_type}.synapse.nml")
    
    def validate_cell_type(self, cell_type):
        """
        Validate that a cell type exists in our predefined list
        
        Args:
            cell_type (str): Cell type to validate
            
        Returns:
            str: Valid cell type or replacement if not found
        """
        if cell_type in self.all_neuron_types:
            return cell_type
        
        # If not found, look for a prefix match
        prefix = cell_type[0] if len(cell_type) > 0 else ''
        if prefix in self.cell_classes:
            return self.cell_classes[prefix][0]  # Return the first of that class
        
        # Default to a common motor neuron if no match
        return "M1"
    
    def export_to_neuroml(self, G, output_path, node_types=None):
        """
        Export a graph to a NeuroML file that matches the C. elegans pharyngeal
        nervous system format
        
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
            
            # Verify that all assigned cell types are valid
            for node, info in node_types.items():
                cell_class = info.get('cell_class', '')
                cell_class = self.validate_cell_type(cell_class)
                node_types[node]['cell_class'] = cell_class
            
            # Assign synapse properties
            synapse_props = self.assign_synapse_properties(G, node_types)
            
            # Create the NeuroML document
            network_id = os.path.splitext(os.path.basename(output_path))[0]
            neuroml_doc = self.create_neuroml_document(network_id)
            
            # Include references to necessary cell and synapse files
            # Sort to ensure consistent order
            cell_types = sorted(set(info['cell_class'] for info in node_types.values()))
            
            # Include all cell types
            for cell_class in cell_types:
                etree.SubElement(neuroml_doc, "include", href=f"{cell_class}.cell.nml")
            
            # Collect all synapse types needed
            synapse_types_used = set()
            gap_junction_needed = False
            
            for props in synapse_props.values():
                synapse_type = props['synapse_type']
                synapse_types_used.add(synapse_type)
                if props['is_gap_junction']:
                    gap_junction_needed = True
            
            # Include standard synapse references
            for synapse_type in sorted(synapse_types_used):
                synapse_file = self.get_synapse_filename(synapse_type)
                etree.SubElement(neuroml_doc, "include", href=synapse_file)
            
            # Add gap junction reference if needed
            if gap_junction_needed and 'Generic_GJ' not in synapse_types_used:
                etree.SubElement(neuroml_doc, "include", href="Generic_GJ.nml")
            
            # Create network container
            network = etree.SubElement(
                neuroml_doc,
                "network",
                id=f"{network_id}",
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
            for cell_class, node_ids in sorted(population_nodes.items()):
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
                etree.SubElement(annotation, "property", tag="color", value=f"{r:.2f} {g:.2f} {b:.2f}")
                
                # Add instances
                for i, node_id in enumerate(node_ids):
                    instance = etree.SubElement(pop, "instance", id=str(i))
                    
                    # Get position if available
                    pos = None
                    if hasattr(node_id, '__iter__'):
                        pos = node_id  # Already a position tuple
                    elif 'pos' in G.nodes.get(node_id, {}):
                        pos = G.nodes[node_id]['pos']
                    else:
                        # Default position in the plane
                        pos = (np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0.0)
                    
                    # Ensure position is a tuple or list with at least 2 elements
                    if not isinstance(pos, (tuple, list)) or len(pos) < 2:
                        pos = (np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0.0)
                    
                    # Add z-coordinate if missing
                    if len(pos) == 2:
                        pos = (pos[0], pos[1], 0.0)
                    
                    # Add location
                    etree.SubElement(
                        instance,
                        "location",
                        x=str(float(pos[0])),
                        y=str(float(pos[1])),
                        z=str(float(pos[2]) if len(pos) > 2 else 0.0)
                    )
            
            # Group connections by type to create projections
            projection_counter = 0
            chemical_projections = {}  # (source_class, target_class, synapse_type) -> projection
            electrical_projections = {} # (source_class, target_class) -> projection
            
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
                    existing_proj = None
                    for ep in network.xpath(f".//electricalProjection"):
                        if ep.get("id") == proj_id:
                            existing_proj = ep
                            break
                            
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
                    pre_segment = np.random.randint(1, 9)
                    post_segment = np.random.randint(1, 9)
                    pre_frac = np.random.random()
                    post_frac = np.random.random()
                    
                    conn.set("preSegment", str(pre_segment))
                    conn.set("postSegment", str(post_segment))
                    conn.set("preFractionAlong", str(pre_frac))
                    conn.set("postFractionAlong", str(post_frac))
                    
                else:
                    # Chemical synapse
                    proj_id = f"NCXLS_{source_class}_{target_class}"
                    
                    # Check if projection already exists
                    existing_proj = None
                    for ep in network.xpath(f".//projection"):
                        if ep.get("id") == proj_id:
                            existing_proj = ep
                            break
                            
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
                    pre_segment = np.random.randint(1, 9)
                    post_segment = np.random.randint(1, 9)
                    pre_frac = np.random.random()
                    post_frac = np.random.random()
                    
                    conn.set("preSegmentId", str(pre_segment))
                    conn.set("postSegmentId", str(post_segment))
                    conn.set("preFractionAlong", str(pre_frac))
                    conn.set("postFractionAlong", str(post_frac))
                
                projection_counter += 1
                        
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
                'cell_class_distribution': {}
            }
            
            # Count cell classes
            cell_classes = [info['cell_class'] for info in node_types.values()]
            class_counts = Counter(cell_classes)
            total_count = len(cell_classes)
            
            for cell_class, count in class_counts.items():
                output_data['cell_class_distribution'][cell_class] = count / total_count
            
            # Add basic type distribution
            basic_types = [info['basic_type'] for info in node_types.values()]
            basic_counts = Counter(basic_types)
            
            output_data['basic_type_distribution'] = {
                t: basic_counts.get(t, 0) / total_count for t in ['motor', 'sensory', 'inter']
            }
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
                
            print(f"Successfully exported cell types to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting cell types: {e}")
            return False
    
    def analyze_neuroml_file(self, neuroml_path):
        """
        Analyze an existing NeuroML file to extract cell classes and connection patterns
        
        Args:
            neuroml_path (str): Path to the NeuroML file
            
        Returns:
            dict: Dictionary with extracted information
        """
        try:
            # Parse the NeuroML file
            tree = etree.parse(neuroml_path)
            root = tree.getroot()
            
            # Extract namespaces
            nsmap = {k if k is not None else 'neuroml': v for k, v in root.nsmap.items()}
            
            # Extract all cell types (populations)
            populations = {}
            for pop in root.xpath('//neuroml:population', namespaces=nsmap):
                pop_id = pop.get('id')
                component = pop.get('component')
                size = pop.get('size')
                
                populations[pop_id] = {
                    'component': component,
                    'size': size
                }
            
            # Extract chemical projections
            chem_projections = {}
            for proj in root.xpath('//neuroml:projection', namespaces=nsmap):
                proj_id = proj.get('id')
                pre_pop = proj.get('presynapticPopulation')
                post_pop = proj.get('postsynapticPopulation')
                synapse = proj.get('synapse')
                
                # Count connections
                connections = proj.xpath('.//neuroml:connection', namespaces=nsmap)
                
                chem_projections[proj_id] = {
                    'pre_pop': pre_pop,
                    'post_pop': post_pop,
                    'synapse': synapse,
                    'connection_count': len(connections)
                }
            
            # Extract electrical projections (gap junctions)
            elect_projections = {}
            for proj in root.xpath('//neuroml:electricalProjection', namespaces=nsmap):
                proj_id = proj.get('id')
                pre_pop = proj.get('presynapticPopulation')
                post_pop = proj.get('postsynapticPopulation')
                
                # Count connections
                connections = proj.xpath('.//neuroml:electricalConnection', namespaces=nsmap)
                
                elect_projections[proj_id] = {
                    'pre_pop': pre_pop,
                    'post_pop': post_pop,
                    'connection_count': len(connections)
                }
            
            # Extract included files
            include_files = []
            for include in root.xpath('//neuroml:include', namespaces=nsmap):
                href = include.get('href')
                include_files.append(href)
            
            # Collect cell classes
            cell_classes = {}
            for include_file in include_files:
                if include_file.endswith('.cell.nml'):
                    cell_name = include_file.split('.cell.nml')[0]
                    
                    # Determine basic type from prefix
                    basic_type = 'unknown'
                    for prefix, class_type in self.prefix_to_type.items():
                        if cell_name.startswith(prefix):
                            basic_type = class_type
                            break
                    
                    cell_classes[cell_name] = {
                        'basic_type': basic_type
                    }
            
            # Update internal cell classes based on the file
            self.all_neuron_types = list(cell_classes.keys())
            
            # Group cell classes by prefix
            prefix_groups = {}
            for cell_name in self.all_neuron_types:
                # Extract prefix (M, I, MC, etc.)
                prefix = None
                for p in self.prefix_to_type.keys():
                    if cell_name.startswith(p):
                        prefix = p
                        break
                
                if prefix:
                    if prefix not in prefix_groups:
                        prefix_groups[prefix] = []
                    prefix_groups[prefix].append(cell_name)
                    
            # Update cell classes
            self.cell_classes = prefix_groups
            
            # Collect all synapse types
            synapse_types = {}
            for include_file in include_files:
                if include_file.endswith('synapse.nml') or include_file.endswith('.nml'):
                    base_name = include_file.split('.')[0]
                    if 'Acetylcholine' in base_name:
                        synapse_types['excitatory'] = base_name
                    elif 'Glutamate' in base_name:
                        synapse_types['inhibitory'] = base_name
                    elif 'Serotonin' in base_name or 'Dopamine' in base_name:
                        synapse_types['modulatory'] = base_name
                    elif 'GJ' in base_name or 'gap' in base_name.lower():
                        synapse_types['gap_junction'] = base_name
            
            # Update synapse types if found
            if synapse_types:
                self.synapse_types.update(synapse_types)
            
            return {
                'populations': populations,
                'chemical_projections': chem_projections,
                'electrical_projections': elect_projections,
                'cell_classes': cell_classes,
                'synapse_types': self.synapse_types
            }
            
        except Exception as e:
            print(f"Error analyzing NeuroML file: {e}")
            return {}
    
    def update_cell_classes_from_directory(self, nml_dir):
        """
        Update cell classes by scanning a directory of NeuroML files
        
        Args:
            nml_dir (str): Path to directory containing NeuroML files
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Scan the directory for cell files
            cell_types = []
            for filename in os.listdir(nml_dir):
                if filename.endswith('.cell.nml'):
                    cell_name = filename.split('.cell.nml')[0]
                    cell_types.append(cell_name)
            
            if not cell_types:
                print(f"No cell files found in {nml_dir}")
                return False
            
            # Update neuron types
            self.all_neuron_types = cell_types
            
            # Group by prefix to update cell classes
            prefix_groups = {}
            
            # Sort the cell types to ensure consistent grouping
            for cell_name in sorted(cell_types):
                # Detect prefix - special handling for multi-character prefixes
                prefix = None
                if cell_name.startswith('NSM'):
                    prefix = 'NSM'
                elif cell_name.startswith('MC'):
                    prefix = 'MC'
                elif cell_name.startswith('MI'):
                    prefix = 'MI'
                elif len(cell_name) > 0:
                    # Default to first character as prefix
                    prefix = cell_name[0]
                
                if prefix:
                    if prefix not in prefix_groups:
                        prefix_groups[prefix] = []
                    prefix_groups[prefix].append(cell_name)
            
            # Filter to only include pharyngeal-related cell types
            pharyngeal_prefixes = {'M', 'I', 'MC', 'MI', 'NSM'}
            self.cell_classes = {k: v for k, v in prefix_groups.items() if k in pharyngeal_prefixes}
            
            # If no pharyngeal cells found, just use all cells but give a warning
            if not self.cell_classes:
                print("Warning: No pharyngeal neuron types found. Using all cell types.")
                self.cell_classes = prefix_groups
            
            # Update prefix to type mapping if needed
            for prefix in self.cell_classes.keys():
                if prefix not in self.prefix_to_type:
                    if prefix in {'M'}:
                        self.prefix_to_type[prefix] = 'motor'
                    elif prefix in {'I'}:
                        self.prefix_to_type[prefix] = 'sensory'
                    else:
                        self.prefix_to_type[prefix] = 'inter'
            
            print(f"Updated cell classes from {nml_dir}: {len(self.all_neuron_types)} cell types in {len(self.cell_classes)} groups")
            return True
            
        except Exception as e:
            print(f"Error updating cell classes from directory: {e}")
            return False
    
    def update_synapse_types_from_directory(self, nml_dir):
        """
        Update synapse types by scanning a directory of NeuroML files
        
        Args:
            nml_dir (str): Path to directory containing NeuroML files
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Scan for synapse files
            synapse_files = []
            for filename in os.listdir(nml_dir):
                if filename.endswith('.synapse.nml') or filename == 'Generic_GJ.nml':
                    synapse_files.append(filename)
            
            if not synapse_files:
                print(f"No synapse files found in {nml_dir}")
                return False
            
            # Map synapse files to types
            synapse_map = {}
            
            for filename in synapse_files:
                base_name = filename.split('.')[0]
                
                # Categorize by names
                if 'Generic_GJ' in filename:
                    synapse_map['gap_junction'] = base_name
                elif 'Acetylcholine' in base_name:
                    if 'Tyramine' in base_name:
                        # Special case for mixed synapse
                        synapse_map['modulatory'] = base_name
                    else:
                        synapse_map['excitatory'] = base_name
                elif 'Glutamate' in base_name:
                    synapse_map['inhibitory'] = base_name
                elif 'Serotonin' in base_name:
                    if 'Glutamate' in base_name or 'Acetylcholine' in base_name:
                        synapse_map['modulatory'] = base_name
                    else:
                        synapse_map['modulatory'] = base_name
                elif any(mod in base_name for mod in ['Dopamine', 'GABA', 'FMRFamide', 'Octapamine']):
                    synapse_map['modulatory'] = base_name
            
            # Update synapse types
            for synapse_type, synapse_name in synapse_map.items():
                self.synapse_types[synapse_type] = synapse_name
            
            print(f"Updated synapse types from {nml_dir}: {self.synapse_types}")
            return True
            
        except Exception as e:
            print(f"Error updating synapse types from directory: {e}")
            return False
    
    def initialize_from_directory(self, nml_dir):
        """
        Initialize cell classes and synapse types from a NeuroML directory
        
        Args:
            nml_dir (str): Path to directory containing NeuroML files
            
        Returns:
            bool: True if successful, False otherwise
        """
        success1 = self.update_cell_classes_from_directory(nml_dir)
        success2 = self.update_synapse_types_from_directory(nml_dir)
        
        return success1 and success2