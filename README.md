# Gestalt Vector Field Theory: Toward a Field-Theoretic Framework for Modular Cognition 
See: https://github.com/im-knots/gvft/blob/main/gvft.html for full overview

This is a heavily work in progress and unpolished repo im working on as a fun side project. Please bear with me

**Gestalt Vector Field Theory (GVFT)** introduces a meta-architectural framework for specifying and evolving modular neural network topologies using continuous multi-field representations defined over a spatial domain. Instead of encoding fixed graphs, GVFT defines smooth, differentiable fields—such as connectivity flows, synaptic strengths, conduction delays, and neuromodulatory gradients—that serve as generative blueprints for network formation. We explore this framework through two experimental regimes: (1) synthetic field simulations to map emergent dynamics across parameter space, and (2) a biologically grounded loop that converts real connectomes from NeuroML2 into GVFT fields, evolves them, and returns them back into NeuroML2 for behavioral testing using pyneuroml neuron simulations. This repository introduces the mathematical formalism, simulation methodology, simulation engine and proposed experimental validation steps that position GVFT as a biologically plausible and computationally flexible foundation for modular neural architecture design.

## ELI5 
Imagine trying to design a brain—not by hand-crafting every connection, but by drawing invisible winds that tell neurons where to grow and how to talk to each other. GVFT is like using weather maps to design neural networks. Instead of building brains with wires and blueprints, we use fields—smooth shapes that guide how neurons connect. These fields describe things like: who should talk to who, how strong the messages are, how fast they travel, and what mood the brain is in. We start with a real worm brain (from NeuroML) and turn it into fields. Then we let those fields evolve—like watching weather patterns shift over time. Finally, we convert those evolved fields back into a simulated brain and check if it still behaves like a real one. The big idea? We're not drawing a map of the brain. We're drawing the rules for how the brain builds itself.

## Experimental Methodology

We’re testing GVFT through three experimental pipelines:

### 1. **Synthetic Field Simulation (Exploratory Regime)**
- We randomly initialize GVFT fields and evolve them over time using reaction-diffusion dynamics.
- These simulations help us understand how stable, structured patterns (like Turing patterns) emerge.
- By sweeping across parameters, we identify regions of the field space that support structured and persistent dynamics.

### 2. **Biologically Grounded Loop (Validation Regime)**
- We start with a real-world connectome (e.g. the *C. elegans* pharyngeal network in NeuroML2).
- This connectome is converted into GVFT fields, effectively "compressing" it into a spatial field representation.
- The fields are then evolved and re-expanded back into a new neural graph.
- This new network is converted back into NeuroML2 and simulated using NEURON via pyNeuroML.
- We stimulate it and observe whether the evolved architecture shows biologically plausible behavior (e.g. oscillations, persistent activity).

### 3. **Random or Flat Priors (Unsupervised Emergence Regime)**
- We initialize GVFT fields with either flat (uniform) values or structured noise rather than biological priors.
- These uninformed initial conditions evolve freely under the same GVFT dynamics.
- We then instantiate the resulting fields into neural networks, simulate them, and test whether spontaneous or emergent structure arises.
- This tests whether GVFT can discover viable network motifs from scratch, not just preserve existing ones.

### Comparison with Baseline Neural Dynamics
- We run parallel NEURON simulations on both the original biological connectome and the GVFT-evolved versions (from both priors and noise).
- We measure neural activity metrics such as membrane potential traces, frequency spectra, and synchrony.
- If the GVFT-evolved networks exhibit oscillations or coherent activity patterns similar to the original, it supports the claim that GVFT preserves or re-discovers key dynamical motifs.
- If differences arise, we analyze whether they are noise, loss of function, or potentially novel stable regimes.

This multi-pronged approach lets us both explore GVFT's generative potential and validate its biological relevance.



## FUNCTIONAL FEATURES
- neuroml file input as biological priors
- gvft field input from flat or randomized priors
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
