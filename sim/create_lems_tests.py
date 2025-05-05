#!/usr/bin/env python3
"""
Create LEMS simulation files for oscillation testing of NeuroML networks.
This script should be run directly from the command line.
"""

import os
import sys
import glob
from pyneuroml.lems import LEMSSimulation
from pyneuroml import pynml

def create_oscillation_test(neuroml_file, duration=500, dt=0.025, 
                           stim_amplitude=0.5, stim_delay=50, stim_duration=400):
    """
    Create a LEMS simulation file for testing oscillations in the network.
    
    Args:
        neuroml_file (str): Path to the NeuroML file
        duration (float): Simulation duration in ms
        dt (float): Time step in ms
        stim_amplitude (float): Amplitude of stimulus current in nA
        stim_delay (float): Delay before stimulus in ms
        stim_duration (float): Duration of stimulus in ms
        
    Returns:
        str: Path to created LEMS file
    """
    try:
        # Make sure the neuroml_file is an absolute path
        neuroml_file = os.path.abspath(neuroml_file)
        
        # Verify the file exists
        if not os.path.exists(neuroml_file):
            print(f"Error: NeuroML file not found: {neuroml_file}")
            return None
        
        # Extract base name for simulation ID
        neuroml_dir = os.path.dirname(neuroml_file)
        neuroml_filename = os.path.basename(neuroml_file)
        base_name = os.path.splitext(neuroml_filename)[0]
        if base_name.endswith('.net'):
            base_name = base_name[:-4]
        
        # Create unique simulation ID
        simulation_id = f"Sim_{base_name}_oscillation_test"
        
        # Get a network ID from the filename
        network_id = base_name.replace('.', '_')
        
        # We need to run the simulation from the directory containing the NeuroML file
        # Change to the directory containing the NeuroML file
        original_dir = os.getcwd()
        os.chdir(neuroml_dir)
        
        # First, generate pulse generator NeuroML files manually
        # We'll create a separate file for each stimulus
        stimulus_files = []
        stim_cells = ["I1L", "I1R", "I2L", "I2R"]
        
        for cell in stim_cells:
            pg_id = f"{cell.lower()}_stim"
            pg_filename = f"{pg_id}.nml"
            stimulus_files.append(pg_filename)
            
            # Create NeuroML pulse generator file content
            pg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://www.neuroml.org/schema/neuroml2 https://raw.github.com/NeuroML/NeuroML2/development/Schemas/NeuroML2/NeuroML_v2beta4.xsd"
         id="{pg_id}">
    
    <pulseGenerator id="{pg_id}" delay="{stim_delay}ms" duration="{stim_duration}ms" amplitude="{stim_amplitude}nA"/>
    
</neuroml>
"""
            # Write to file
            with open(pg_filename, 'w') as f:
                f.write(pg_content)
            
            print(f"Created pulse generator file: {pg_filename}")
        
        # Define colors for visualization
        colors = ["#ff0000", "#00ff00", "#0000ff", "#ff00ff", "#00ffff", "#ffff00", "#990099"]
        
        # Motor neuron populations to monitor
        motor_neurons = ["M1", "M2L", "M2R", "M3L", "M3R", "M4", "M5"]
        
        # Create the LEMS simulation file using PyNeuroML API
        lems_file = f"LEMS_{simulation_id}.xml"
        
        # Use the PyNeuroML API to create the LEMS file
        # The LEMSSimulation constructor takes simulation_id, duration, timestep, target, and seed
        simulation = LEMSSimulation(simulation_id, duration, dt, network_id, "12345")
        
        # Include the NeuroML model file
        simulation.include_neuroml2_file(neuroml_filename)
        
        # Include the generated pulse generator files
        for pg_file in stimulus_files:
            simulation.include_neuroml2_file(pg_file)
        
        # Create a display for visualizing membrane potentials
        simulation.create_display("display_voltages", "Membrane Potentials", "-80", "40")
        
        # Add motor neuron traces to display
        for i, neuron in enumerate(motor_neurons):
            color = colors[i % len(colors)]
            simulation.add_line_to_display("display_voltages", f"{neuron}_v", f"{neuron}/0/{neuron}/v", "1mV", color)
        
        # Create output file for data
        simulation.create_output_file("output_voltages", f"{base_name}_voltages.dat")
        
        # Add outputs for different neuron types
        # Motor neurons
        for neuron in motor_neurons:
            simulation.add_column_to_output_file("output_voltages", f"{neuron}_v", f"{neuron}/0/{neuron}/v")
        
        # Sensory neurons
        for neuron in ["I1L", "I1R", "I2L", "I2R"]:
            simulation.add_column_to_output_file("output_voltages", f"{neuron}_v", f"{neuron}/0/{neuron}/v")
        
        # Interneurons
        for neuron in ["I3", "I4", "I5", "I6", "MI"]:
            simulation.add_column_to_output_file("output_voltages", f"{neuron}_v", f"{neuron}/0/{neuron}/v")
        
        # We need to create InputLists manually, but the API doesn't have a direct method for this
        # We'll generate the complete input list manually after creating the core LEMS file
        
        # Save the initial version of the LEMS file
        lems_file_temp = simulation.save_to_file()
        
        # Now we need to manually add the InputList elements for the stimuli
        # Read the saved file
        with open(lems_file_temp, 'r') as f:
            lems_content = f.read()
        
        # Find the position to insert InputList elements - right before the Simulation element
        simulation_pos = lems_content.find("<Simulation ")
        if simulation_pos == -1:
            print("Warning: Could not find Simulation element in LEMS file. InputLists may not be added correctly.")
            simulation_pos = lems_content.find("</Target>") + 10  # Default position after Target
        
        # Generate InputList XML for each stimulus
        input_list_xml = ""
        for cell in stim_cells:
            pg_id = f"{cell.lower()}_stim"
            input_list_xml += f"""
    <InputList id="input_{pg_id}" component="{pg_id}" population="{cell}">
        <input id="0" target="../{cell}/0/{cell}" destination="synapses"/>
    </InputList>"""
        
        # Insert InputList elements into the LEMS file content
        modified_lems_content = lems_content[:simulation_pos] + input_list_xml + "\n    " + lems_content[simulation_pos:]
        
        # Write the modified content back to the file
        with open(lems_file, 'w') as f:
            f.write(modified_lems_content)
        
        print(f"Created LEMS file: {lems_file}")
        
        # Validate the file with PyNeuroML
        print("Validating LEMS file...")
        try:
            validate_result = pynml.validate_lems_file(lems_file)
            if validate_result:
                print("LEMS file validation successful!")
            else:
                print("WARNING: LEMS file validation failed. Manual validation recommended.")
        except Exception as validate_error:
            print(f"Error during validation: {validate_error}")
        
        # Change back to the original directory
        os.chdir(original_dir)
        
        # Return the full path to the LEMS file
        full_lems_path = os.path.join(neuroml_dir, lems_file)
        print(f"Created LEMS simulation file: {full_lems_path}")
        print(f"You can run this simulation with: cd {neuroml_dir} && pynml {lems_file}")
        return full_lems_path
    
    except Exception as e:
        print(f"Error creating LEMS simulation file: {e}")
        import traceback
        traceback.print_exc()
        # Make sure we change back to the original directory in case of an error
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return None

def main():
    """Process command line arguments or use default path."""
    # Get directory or file path from command line
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isdir(path):
            # If directory, process all .net.nml files (not all .nml files, which include cell definitions)
            files = glob.glob(os.path.join(path, "*network*.nml")) + \
                   glob.glob(os.path.join(path, "*.net.nml"))
            # If no network files found, try other nml files that might be networks
            if not files:
                print("No network files found, looking for other .nml files...")
                files = glob.glob(os.path.join(path, "*.nml"))
        elif os.path.isfile(path):
            # If file, process just that file
            files = [path]
        else:
            print(f"Error: Path not found: {path}")
            return
    else:
        # Default directory
        path = "../figures/neuroml"
        if os.path.isdir(path):
            files = glob.glob(os.path.join(path, "*network*.nml")) + \
                   glob.glob(os.path.join(path, "*.net.nml"))
            if not files:
                files = glob.glob(os.path.join(path, "*.nml"))
        else:
            print(f"Error: Default directory not found: {path}")
            print("Please specify a directory or file path.")
            return
    
    if not files:
        print(f"No NeuroML files found in {path}")
        return
    
    print(f"Found {len(files)} NeuroML files")
    
    # Process each file
    for nml_file in files:
        # Skip files that look like cell definitions
        if ".cell.nml" in nml_file.lower():
            print(f"Skipping cell definition file: {nml_file}")
            continue
            
        # Skip synapse files
        if ".synapse.nml" in nml_file.lower():
            print(f"Skipping synapse definition file: {nml_file}")
            continue
        
        print(f"Creating oscillation test for: {nml_file}")
        lems_file = create_oscillation_test(nml_file)
        if lems_file:
            print(f"Success! Run with: cd {os.path.dirname(lems_file)} && pynml {os.path.basename(lems_file)}")
            print("-" * 40)
        else:
            print(f"Failed to create test for: {nml_file}")
            print("-" * 40)

if __name__ == "__main__":
    main()