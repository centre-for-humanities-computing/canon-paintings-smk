# create virtual environment
python3 -m venv env

# activate env
source ./env/bin/activate

# install required packages
pip install -r requirements.txt

# umap-learn causes a lot of pain with dependency issues, fix by installing independently with no set package version here
pip install umap-learn

# explicitly install ipykernel as well
pip install ipykernel

# install kernel so env can be used in jupyter notebooks
python -m ipykernel install --user --name=canon_env

deactivate 