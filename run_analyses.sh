source ./env/bin/activate

# create plots and classification reports
#python3 src/run_analyses.py --df_name df_new_prod_years.pkl --plot --classification --results_suffix new_start_year

export PYTHONPATH=$(pwd)

python3 src/run_analyses.py --df_name smk_canon_paintings.pkl --ds_name smk_canon_paintings --plot --classification

# don't run classification
#python3 src/run_analyses.py --df_name df_filtered_nationalities.pkl --plot --no-classification 

#python3 -m src.paper_plots --df_name df_new_prod_years.pkl --ds_name smk_only_paintings_NEW_DATES

deactivate