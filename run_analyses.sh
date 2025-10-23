source ./env/bin/activate

# necessary in order to use utils scripts
export PYTHONPATH=$(pwd)

python3 src/run_analyses.py --ds_name chcaa/smk_canon_paintings --plot --classification

deactivate