# Gestalt Vector Field Theory: Toward a Field-Theoretic Framework for Modular Cognition 
See: https://github.com/im-knots/gvft/blob/main/gvft.html for full overview
Gestalt Vector Field Theory (GVFT) introduces a meta-architectural framework for specifying and evolving modular neural network topologies using continuous multi-field representations defined over a spatial domain. Instead of encoding fixed graphs, GVFT defines smooth, differentiable fields—such as connectivity flows, synaptic strengths, conduction delays, and neuromodulatory gradients—that serve as generative blueprints for network formation. We explore this framework through two experimental regimes: (1) synthetic field simulations to map emergent dynamics across parameter space, and (2) a biologically grounded loop that converts real connectomes from NeuroML2 into GVFT fields, evolves them, and returns them back into NeuroML2 for behavioral testing using pyneuroml neuron simulations. This repository introduces the mathematical formalism, simulation methodology, simulation engine and proposed experimental validation steps that position GVFT as a biologically plausible and computationally flexible foundation for modular neural architecture design.

## FUNCTIONAL FEATURES
- neuroml file input as biological priors
- gvft field interation/evolution simulation engine
- gfvt field to neuroml instantiation
- neuroml to LEM simulation file
- LEM simulation file to NEURON simulation

## TODO
- FIX: Issue with input and output cells in simulation not getting any inputs.

## Local Env setup
```bash
python3 -m venv gvft-env
source gvft-env/bin/activate 
pip install -r requirements.txt

#run gvft evolution sim with biological priors (natural worm brain input)
cd sim
python3 main.py --neuroml-file ../organicNeuroML2/PharyngealNetwork.net.nml


# Validation 
# Copy a selected generated .net.nml file to the organicNeuroML2 directory then run something like this
pynml -validate ../organicNeuroML2/gvft_network_bio_prior_lamW_0.144_DF_0.019_t20.net.nml

# Create a LEMS simulation file
python create_lems_tests.py ../organicNeuroML2/

# Run LEMS simulation file with neuron interpreter 
pynml ../organicNeuroML2/LEMS_Sim_gvft_network_bio_prior_lamW_0.144_DF_0.019_t20_oscillation_test.xml -neuron -outputdir ../neuron-sim-data
cd ../neuron-sim-data
nrnivmodl # Compile the .mod files
python3 LEMS_Sim_gvft_network_bio_prior_lamW_0.144_DF_0.019_t50_oscillation_test_nrn.py

# Run the simulation
pynml LEMS_Sim_gvft_network_pharyngeal.xml -nogui
```
