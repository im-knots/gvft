## Local Env setup
```bash
python3 -m venv gvft-env
source gvft-env/bin/activate 
pip install -r requirements.txt

#gen priors from neuroml2 data
python3 neuroml_to_gvft.py /home/knots/git/CElegansNeuroML/CElegans/generatedNeuroML2 /home/knots/git/gvft/source-fields

#run sim
cd sim
python3 main.py --neuroml-fields ../source-fields --neuroml-basename PharyngealNetwork

```
