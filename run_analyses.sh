source ./env/bin/activate

export PYTHONPATH=$(pwd)

python3 src/run_analyses.py --df_name smk_canon_paintings.pkl --ds_name smk_canon_paintings --plot --classification

deactivate