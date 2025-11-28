# Framing the Canon: A Computational Study of Canonicity in Danish Golden Age Paintings (1750-1870)

<a href="https://chc.au.dk"><img src="https://github.com/centre-for-humanities-computing/intra/raw/main/images/onboarding/CHC_logo-turquoise-full-name.png" width="25%" align="right"/></a>
 
This repository contains the code to reproduce results from our paper:

"Framing the Canon: A Computational Study of Canonicity in Danish Golden Age Paintings (1750-1870)"

## Citation

Please cite our paper if you use the code or the embeddings: 

```
@article{hansen_framing_2025,
  title = {Framing the Canon: A Computational Study of Canonicity in Danish Golden Age Paintings (1750-1870)},
  author = {Louise Brix Pilegaard Hansen and Rie Schmidt Eriksen and Pascale Feldkamp and Alie Lassche and Kristoffer Nielbo and Katrine Baunvig and Yuri Bizzoni},
  year = {2025},
  journal = {Anthology of Computers and the Humanities},
  volume = {3},
  pages = {339--356},
  editor = {Taylor Arnold, Margherita Fantoli, and Ruben Ros},
  doi = {10.63744/KTLpQIY247dD}
}

```

## Data & Code

### Data
Dataset is fetched from SMKs digital collection using the [SMK-API](https://www.smk.dk/en/article/smk-api/). The dataset used for this analysis can be found on [HuggingFace](https://huggingface.co/datasets/chcaa/smk_canon_paintings).

It should be noted that SMK frequently updates their collection, meaning that there could appear changes in the SMK-API that are not in our dataset, for example to the SMK-IDs or the image URLs. Therefore, pay attention if any images fail when downloading the image files from the URLs.

### Code

- ```src/analyses_utils.py``` and ```src/run_analyses.py``` contain the code to run the full analysis on the canon-paintings data. The scripts can be run with set arguments with ```run_analyses.sh``` and output is in the ```results/``` folder
- ```nbs/canon_analysis.ipynb``` runs the same functions as ```src/run_analyses.py``` but in a notebook style with added, descriptive markdown chunks and the option to run each step of the analysis seperately.
- ```src/greyscale_embeddings.py``` and ```src/smk_embeddings.py``` contains the code to extract colored and greyscaled embeddings for the dataset of paintings. These are already in the dataset uploaded to [HuggingFace](https://huggingface.co/datasets/louisebrix/smk_canon_paintings).

#### Running the code
All code was run using macOS 15.6.1 in Visual Studio Code with Python 3.13.3.

To run the code, first set up the virtual environment and install required packages with: 
```
bash setup.sh
```

Next, run analysis with default arguments with:
```
bash run_analyses.sh 
```
Results are in the ```/results``` folder.

## Project Structure

```
canon-paintings-smk/
│
├── nbs/                          # Jupyter Notebooks
│   └── canon_analysis.ipynb      # Code to analyses in notebook instead of script

│
├── results/                      # Output of analysis
│   ├── classification/           # Classification reports from supervised classification
│   └── figs/                     # Output visualizations      
│
├── src/                          # Source code and helper functions
│   ├── analyses_utils.py         # Helper functions to run canon analyses
│   ├── greyscale_embeddings.py   # Extracts greyscaled image embeddings
│   ├── run_analyses.py           # Run full canon analysis
│   └── smk_embeddings.py         # Extract colored image embeddings
│
├── README.md
├── requirements.txt              # Python dependencies
├── run_analyses.sh               # Run canon analyses with set arguments
└── setup.sh                      # Set up virtual environment and install required packages
```