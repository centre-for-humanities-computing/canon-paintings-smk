# Framing the Canon: A Computational Study of Canonicity in Danish Golden Age Paintings (1750-1870)

<a href="https://chc.au.dk"><img src="https://github.com/centre-for-humanities-computing/intra/raw/main/images/onboarding/CHC_logo-turquoise-full-name.png" width="25%" align="right"/></a>

### 

This repository contains the code to reproduce results from our paper:

"Framing the Canon: A Computational Study of Canonicity in Danish Golden Age Paintings (1750-1870)"

## Structure

your-project/
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




Dataset was created by running .. ... ..

The final dataset used for the paper can be found on HuggingFace .. and as a pickle file under /data