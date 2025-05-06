python3 -m venv env

source ./env/bin/activate

pip install -r requirements.txt

pip install ipykernel

python -m ipykernel install --user --name=env

deactivate 