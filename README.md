# Framing the Canon: A Computational Study of Canonicity in Danish Golden Age Paintings (1750-1870)

<a href="https://chc.au.dk"><img src="https://github.com/centre-for-humanities-computing/intra/raw/main/images/onboarding/CHC_logo-turquoise-full-name.png" width="25%" align="right"/></a>
 

This repository contains the code to reproduce results from our paper:

"Framing the Canon: A Computational Study of Canonicity in Danish Golden Age Paintings (1750-1870)"

## Citation

Please cite our paper if you use the code or the embeddings: 

```
add bibtex_formatted_ref

```

## Data & Code

### Data

Dataset with metadata, images and embeddings can be found on [HuggingFace](https://huggingface.co/datasets/louisebrix/smk_canon_paintings).

Full datafolder is not available here due to size limits but can be made available upon request (it is necessary in order to run the full analysis pipeline).

### Code

- ```nbs/create_canon_data.ipynb``` contains code to create the final datafile used for the analysis
- ```src/analyses_utils.py``` and ```src/run_analyses.py``` contains the code to run the full analysis on the canon-paintings data. The scripts are run with set arguments with ```run_analyses.sh``` and output is in the ```results/``` folder
- ```nbs/canon_analysis.ipynb``` runs the same functions as ```src/run_analyses.py``` but with added, descriptive markdown chunks and the option to run each step of the analysis seperately.
- ```src/greyscale_embeddings.py``` and ```src/smk_embeddings.py``` contains the code to extract colored and greyscaled embeddings for the dataset of paintings. These are already in the dataset uploaded to [HuggingFace](https://huggingface.co/datasets/louisebrix/smk_canon_paintings).

## Structure

```
canon-paintings-smk/
│
├── figs/                         # Visualizations for the article
│   ├── ...
│   ├── ...
│   └── ...
│
├── nbs/                          # Jupyter Notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_evaluation.ipynb
│   └── ...
│
├── src/                          # Source code and helper functions
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model_utils.py
│   ├── analysis_pipeline.py
│   └── ...
│
├── run_analyses.sh               # Run all analyses
├── setup.sh                      # Set up virtual environment
├── requirements.txt              # Python dependencies
├── upload_data_to_hub.sh         # Upload SMK dataset to Hugging Face Hub
├── README.md
├── LICENSE
└── CITATION.cff
```