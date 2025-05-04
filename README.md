## Local Env setup
```bash
python3 -m venv gvft-env
source gvft-env/bin/activate 
pip install -r requirements.txt

#run sim
cd sim
python3 main.py --neuroml-file ../organicNeuroML2/PharyngealNetwork.net.nml


# Validation 
# Copy a selected generated .net.nml file to the organicNeuroML2 directory then run something like this
pynml -validate ../organicNeuroML2/gvft_network_bio_prior_lamW_0.144_DF_0.013_t40.net.nml

# Network analysis
python -m pyneuroml.analysis.network_analyzer ../figures/neuroml/gvft_network_bio_prior_lamW_0.144_DF_0.013_t70.net.nml -nc

# Create a LEMS simulation file
pynml-modchan -nml ../figures/neuroml/gvft_network_fixed.net.nml -lems

# Run the simulation
pynml LEMS_Sim_gvft_network_pharyngeal.xml -nogui
```
