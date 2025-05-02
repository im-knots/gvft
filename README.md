## Local Env setup
```bash
python3 -m venv gvft-env
source gvft-env/bin/activate 
pip install -r requirements.txt
python3 gvft_sweep_sim_gpu.py
python3 neuroml_to_gvft.py /home/knots/git/CElegansNeuroML/CElegans/generatedNeuroML2 /home/knots/git/gvft/source-fields
```
