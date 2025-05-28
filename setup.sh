# create virtual environment
python3 -m venv env

# activate env
source ./env/bin/activate

# install requirements
pip install -r requirements.txt

# explicitly install ipykernel
pip install ipykernel

# install kernel so env can be used in jupyter notebooks
python -m ipykernel install --user --name=canon_env

deactivate 