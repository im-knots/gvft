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
