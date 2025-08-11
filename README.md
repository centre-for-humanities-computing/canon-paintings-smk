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

- ```nbs/create_canon_data.ipynb``` contains code to define canon variables, clean the data and to create the final datafile used for the analysis
- ```src/analyses_utils.py``` and ```src/run_analyses.py``` contains the code to run the full analysis on the canon-paintings data. The scripts can be run with set arguments with ```run_analyses.sh``` and output is in the ```results/``` folder
- ```nbs/canon_analysis.ipynb``` runs the same functions as ```src/run_analyses.py``` but in a notebook style with added, descriptive markdown chunks and the option to run each step of the analysis seperately.
- ```src/greyscale_embeddings.py``` and ```src/smk_embeddings.py``` contains the code to extract colored and greyscaled embeddings for the dataset of paintings. These are already in the dataset uploaded to [HuggingFace](https://huggingface.co/datasets/louisebrix/smk_canon_paintings).

## Project Structure

```
canon-paintings-smk/
│
├── nbs/                          # Jupyter Notebooks
│   ├── canon_analysis.ipynb      # Code to analyses in notebook instead for script
│   └── create_canon_data.ipynb   # Contruct canon variables and clean up data
│
├── results/                      # Output of analysis
│   ├── classification/           # Classification reports from supervised classification
│   └── figs/                     # Output visualizations      
│
├── src/                          # Source code and helper functions
│   ├── analyses_utils.py         # Helper functions to run canon analyses
│   ├── greyscale_embeddings.py.  # Extracts greyscaled image embeddings
│   ├── run_analyses.py           # Run full canon analysis
│   ├── smk_embeddings.py         # Extract colored image embeddings
│   └── utils.py                  # Miscellaneous helper functions
│
├── README.md
├── embeddings.sh                 # Extract colored and greyscaled embeddings from data
├── requirements.txt              # Python dependencies
├── run_analyses.sh               # Run canon analyses with set arguments
└── setup.sh                      # Set up virtual environment and install required packages
```