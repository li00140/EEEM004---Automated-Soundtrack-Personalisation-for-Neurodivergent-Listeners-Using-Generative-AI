# EEEM004 Accessibility Audio Pipeline

## Setup

### Main environment

python3 -m venv venv
source venv/bin/activate

pip install -r requirements_main.txt

### AudioLDM environment

python3 -m venv audioldm_env
source audioldm_env/bin/activate

pip install -r requirements_audioldm.txt

deactivate

## Run

python3 full_pipeline.py \
  --input_dir . \
  --clip_name testclip \
  --device cpu