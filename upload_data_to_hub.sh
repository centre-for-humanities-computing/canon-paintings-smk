source ./env/bin/activate

pip install huggingface_hub

huggingface-cli login

python3 src/upload_to_hub.py --dataset smk_only_paintings --hub_name smk_only_paintings

deactivate