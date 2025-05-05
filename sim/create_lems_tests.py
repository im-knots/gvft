#!/usr/bin/env python3
"""
Create LEMS simulation files for oscillation testing of NeuroML networks.
This script should be run directly from the command line.
"""

import os
import sys
import glob
from pyneuroml.lems import LEMSSimulation

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
        
        # Create the LEMS simulation
        sim = LEMSSimulation(simulation_id, duration, dt, network_id)
        
        # Include the network file - use just the filename, not the full path
        # since we're now in the same directory
        sim.include_neuroml2_file(neuroml_filename)
        
        # Define a current pulse stimulus for multiple sensory neurons
        # I1 neurons (left and right)
        sim.add_current_input("i1l_stim", stim_amplitude, stim_delay, stim_duration, "I1L", "0")
        sim.add_current_input("i1r_stim", stim_amplitude, stim_delay, stim_duration, "I1R", "0")
        
        # I2 neurons (left and right)
        sim.add_current_input("i2l_stim", stim_amplitude, stim_delay, stim_duration, "I2L", "0")
        sim.add_current_input("i2r_stim", stim_amplitude, stim_delay, stim_duration, "I2R", "0")
        
        # Create a display for visualizing the membrane potentials
        display_id = "display_voltages"
        sim.create_display(display_id, "Membrane Potentials", "-80", "40")
        
        # Motor neuron populations to monitor
        motor_neurons = ["M1", "M2L", "M2R", "M3L", "M3R", "M4", "M5"]
        
        # Define colors for visualization
        colors = ["#ff0000", "#00ff00", "#0000ff", "#ff00ff", "#00ffff", "#ffff00", "#990099"]
        
        # Add motor neurons to display
        for i, neuron in enumerate(motor_neurons):
            color = colors[i % len(colors)]
            sim.add_line_to_display(display_id, f"{neuron}_v", f"{neuron}/0/{neuron}/v", "1mV", color)
        
        # Create output file for data analysis
        output_id = "output_voltages"
        output_file = f"{base_name}_voltages.dat"
        sim.create_output_file(output_id, output_file)
        
        # Add columns to output file - motor neurons
        for neuron in motor_neurons:
            sim.add_column_to_output_file(output_id, f"{neuron}_v", f"{neuron}/0/{neuron}/v")
        
        # Also monitor sensory neurons
        sensory_neurons = ["I1L", "I1R", "I2L", "I2R"]
        for neuron in sensory_neurons:
            sim.add_column_to_output_file(output_id, f"{neuron}_v", f"{neuron}/0/{neuron}/v")
        
        # Interneurons are also interesting to monitor
        interneurons = ["I3", "I4", "I5", "I6", "MI"]
        for neuron in interneurons:
            sim.add_column_to_output_file(output_id, f"{neuron}_v", f"{neuron}/0/{neuron}/v")
        
        # Save LEMS file in the same directory as the NeuroML file
        lems_file = f"LEMS_{simulation_id}.xml"
        sim.save_to_file(lems_file)
        
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