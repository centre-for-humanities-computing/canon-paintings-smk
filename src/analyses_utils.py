'''
Util functions for analyses of canonicity in SMK dataset
'''

# standard libraries
import os
import warnings
from collections import Counter
from random import sample
import requests

# data handling & stats
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
from datasets import Dataset

# visualization tools
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from umap import UMAP
from beautifultable import BeautifulTable

# image processing
from PIL import Image
from datasets import Image as Image_ds # change name because of similar PIL module

# ML
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

# we get a lot of annoying warnings from sklearn so we suppress them
import warnings
warnings.filterwarnings('ignore')

# default plt params
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 14,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.linewidth': 1,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial']
})

################################## PCA ##################################

def pca_binary(ax: plt.Axes, df:pd.DataFrame, embedding:str, canon_category:str, title:str) -> None:

    '''
    Create PCA scatterplot of painting embeddings for each canon category
        
    Parameters:
    - ax: Matplotlib Axes object to plot on
    - df: DataFrame containing the data
    - embedding: Column name with embedding vectors
    - canon_category: Column name for the binary canon category
    - title: Plot title

    '''
    embeddings_array = np.array(df[embedding].to_list(), dtype=np.float32)
    
    color_mapping = {'other': '#129525', 'canon': '#75BCC6'}
    label_mapping = {'other': 'Non-canon', 'canon': 'Canon'}

    # to 2 dimensions
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(embeddings_array)
    df_pca = pd.DataFrame(pca_results, columns=["PCA1", "PCA2"])
    df_pca["canon"] = df[canon_category].values

    # Plot each category
    for category in df_pca["canon"].unique():
        subset = df_pca[df_pca["canon"] == category]
        
        ax.scatter(
            subset["PCA1"],
            subset["PCA2"],
            color=color_mapping.get(category),
            label=label_mapping.get(category),
            alpha=0.6,
            edgecolor='black',
            s=110,
            marker='o'
        )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("PCA1", fontsize=10)
    ax.set_ylabel("PCA2", fontsize=10)
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(True, linestyle='--', alpha=0.2)

    legend_handles = [Patch(facecolor=color_mapping[key], label=label_mapping[key]) for key in color_mapping]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8)

    ax.axis("equal")

    # supress warnings 
    np.seterr(divide='ignore', invalid='ignore')


################################## DIACHRONIC CHANGE ##################################

def calc_bootstrap_CI_intra(n_bootstraps:int, similarities:np.ndarray) -> list:

    '''
    Calculate bootstrapped 95% confidence intervals for mean intragroup similarity

    Parameters:
        - n_bootstraps: Number of bootstrap samples to generate
        - similarities: Array of cosine similarities for in-group embeddings

    Returns:
        - List of lower and upper confidence interval bounds
    '''

    # create random number generator
    rng = np.random.default_rng(seed=42)
    means = []

    for _ in range(n_bootstraps):
        sample = rng.choice(similarities, size=len(similarities), replace=True) # draw samples with replacement from cosine similarities
        means.append(np.mean(sample)) # compute mean of bootstrapped sample
        
    # get confidence intervals
    lower = np.percentile(means, 2.5)
    upper = np.percentile(means, 97.5)

    CI = [lower, upper]

    return CI 

def get_cosim_mean_std(groups_dict:dict, embedding_col:str, key:str) -> tuple:

    '''
    Calculate mean and SD cosine similarity for embedding column of dataset in groups_dict

    Parameters:
        - groups_dict: Dict mapping group keys to DataFrames with embeddings
        - embedding_col: Name of column with image embeddings
        - key: Name of group to compute cosine similarity for (i.e., canon/non-canon)
    
    Returns:
        - mean_cosim: Mean of cosine similarity in group
        - CI: Upper and lower bounds of 95% bootstrapped confidence interval
    '''

    # get group
    data = groups_dict[key]

    # convert from list of arrays to single numpy array with added dim
    embeddings = np.stack(data[embedding_col].values)

    # get matrix of cosine sim pairs
    similarities = cosine_similarity(embeddings)

    # calc mean cosim
    mean_cosim = similarities.mean()

    # calc bootstrapped confidence intervals
    CI = calc_bootstrap_CI_intra(1000, similarities)
    
    return mean_cosim, CI

def create_groups(df:pd.DataFrame, year_col:str, canon_col:str, year_range:list) -> dict:

    '''
    Create dict with groups of canon, non-canon and total paintings in specified time window

    Parameters:
        - df: Dataframe with SMK data
        - year_col: Name of column with prod years
        - canon_col: Canon variable to use
        - year_range: List of years in window
    
    Returns:
        - groups: Dict with canon, non-canon and total dataframes 
    '''
    
    # filter dataframes based on year range and canon status
    canon = df.loc[(df[year_col].isin(year_range)) & (df[canon_col] == 'canon')]
    df_total = df.loc[df[year_col].isin(year_range)]
    non_canon = df.loc[(df[year_col].isin(year_range)) & (df[canon_col] == 'other')]

     # create dict
    groups = {'canon': canon, 'df_total': df_total, 'non_canon': non_canon}

    return groups

def get_all_cosims(df:pd.DataFrame, year_col:str, canon_col:str, year_range:list, embedding_col:str, sampling:bool, sample_size:int, run:int) -> dict:

    '''
    Calculate all cosine similarity measures for current time window and return as dict. 

    First creates groups of canon, non-canon and total paintings in the current time window.
    Next, it calculates the mean cosine similarity of the embeddings in the current window for each group (intra-group).
    The cosine similarity between the mean canon and non-canon embeddings is also calculated (inter-group).

    Parameters: 
        - df: Dataframe with smk data
        - year_col: Name of column with prod years
        - canon_col: Canon variable to use
        - year_range: List of years in window
        - embedding_col: Name of column in df containing embeddings
        - sampling: Whether to sample groups randomly
        - sample_size = Number of samples
        - run: Random state for sampling
    
    Returns:
        - temp: dict with cosine similarity measures, confidence intervals and counts for each group
    '''

    # get dict with filtered canon, non-canon and total dataframes in window
    groups = create_groups(df, year_col, canon_col, year_range)

    if sampling == True:
        for key in groups:
            group = groups[key]
            groups[key] = group.sample(sample_size, random_state=run) if len(group) > sample_size else group

    # get the mean embeddings of the current window for each group
    canon_mean = groups['canon'][embedding_col].mean(axis=0)
    non_canon_mean = groups['non_canon'][embedding_col].mean(axis=0)

    temp = {} 

    # get the mean cosine similarity between mean canon embedding and mean non-canon embedding for this time window (inter-group)
    canon_noncanon_similarity = cosine_similarity(np.stack([non_canon_mean, canon_mean]))[0][1]
    temp['CANON_NONCANON_COSIM'] = canon_noncanon_similarity

    # get mean cosine similarity of canon embeddings for this time window + confidence interval
    canon_mean, canon_CI = get_cosim_mean_std(groups, embedding_col, 'canon')
    temp['CANON_COSIM_MEAN'] = canon_mean 
    temp['CANON_COSIM_CI'] = canon_CI

    # get mean cosine similarity of non-canon embeddings for this time window + confidence interval
    nc_mean, nc_CI = get_cosim_mean_std(groups, embedding_col, 'non_canon')
    temp['NONCANON_COSIM_MEAN'] = nc_mean
    temp['NONCANON_COSIM_CI'] = nc_CI

    # get mean cosine similarity of all data for this time window + confidence interval
    t_mean, t_CI = get_cosim_mean_std(groups, embedding_col, 'df_total')
    temp['TOTAL_COSIM_MEAN'] = t_mean
    temp['TOTAL_COSIM_CI'] = t_CI

    # add counts of paintings in group to dict
    temp['n_paintings'] = [len(groups['df_total']), len(groups['canon']), len(groups['non_canon'])]
    
    return temp

def run_change_analysis(w_size: int,
                        df: pd.DataFrame,
                        canon_col: str,
                        embedding_col: str,
                        step_size: int = 1,
                        year_col: str = 'start_year',
                        n_runs: int = 1,
                        sampling: bool = False,
                        sample_size: int = 0) -> pd.DataFrame:
    
    '''
    Performs rolling window analyses over years to compute cosine similarity measures

    Parameters:
        - w_size: Size of rolling window in years
        - df: SMK dataframe with embeddings
        - canon_col: Canon variable to use
        - embedding_col: Name of column in df containing embeddings
        - step_size: Step size between windows
        - year_col: Name of column with prod years
        - n_runs: Number of runs for sampling
        - sampling: Whether to sample paintings in groups
        - sample_size: Number of samples if sample is true

    Returns:
        - sim_df: DataFrame with rolling window cosine similarity results
    '''

    # Start a loop over the years
    mean_similarity_dict = {}

    # Get the minimum and maximum years in the dataset
    min_year = df[year_col].min()
    max_year = df[year_col].max()

    for run in range(n_runs):
        for start_year in range(min_year, max_year - w_size + 1, step_size):

            # Define rolling window range for each window
            year_range = list(range(start_year, start_year + w_size))
            range_label = f"{year_range[0]}-{year_range[-1]}"

            # calculate cosine similarity measures
            temp = get_all_cosims(df, year_col, canon_col, year_range, embedding_col, sampling, sample_size, run)
            
            # add cosine similarity measure to dict for dict at time window
            mean_similarity_dict[range_label] = temp
    
    # create dataframe from dict
    sim_df = pd.DataFrame.from_dict(mean_similarity_dict, orient='index').reset_index()
    sim_df = sim_df.rename(columns={"index": "year_RANGE"})

    # add start year column
    sim_df['START_year'] = sim_df['year_RANGE'].apply(lambda x: int(x.split('-')[0]))

    return sim_df

def plot_diachronic_change(w_size: int,
                           df: pd.DataFrame,
                           canon_col: str,
                           embedding_col: str,
                           cosim_to_plot: str,
                           ax: plt.Axes,
                           step_size: int = 1,
                           year_col: str = 'start_year',
                           n_runs: int = 1,
                           sampling: bool = False,
                           sample_size: int = 0,
                           color: str = 'C0'):
    
    '''
    Create lineplot showing rolling window analysis of cosine similarities

    Parameters:
        - w_size: Size of rolling window in years
        - df: SMK dataframe with embeddings
        - canon_col: Canon variable to use
        - embedding_col: Name of column in df containing embeddings
        - cosim_to_plot: Cosine Similarity measure to plot
        - ax: plt axes to plot on
        - step_size: Step size between windows
        - year_col: Name of column with prod years
        - n_runs: Number of runs for sampling
        - sampling: Whether to sample paintings in groups
        - sample_size: Number of samples if sample is true
        - color: Default line color for the plot
        '''

    # run rolling window analysis
    sim_df = run_change_analysis(
                w_size,
                df,
                canon_col,
                embedding_col,
                step_size=step_size,
                year_col=year_col,
                n_runs=n_runs,
                sampling=sampling,
                sample_size=sample_size,
    )
    
    # get correlation between year and cosine similarity
    corr, pval = spearmanr(sim_df['START_year'], sim_df[cosim_to_plot])

    # if greyscaled embeddings, make lineplot grey
    if embedding_col == 'grey_embedding':
        color = 'grey'

    # plot change over time
    ax.plot(sim_df['START_year'], 
            sim_df[cosim_to_plot], 
            color=color, 
            linewidth=3, 
            alpha=1)

    # fix titles for plot
    title_mapping = {'exb_canon': 'Exhibitions canon',
                     'smk_exhibitions': 'SMK exhibitions canon',
                     'on_display': 'On display canon'}

    col_or_grey = 'colored' if embedding_col == 'embedding' else 'greyscaled'

    ylabel = 'Mean Cosine Similarity'

    # add specifics of plot based on cosim to plot
    if cosim_to_plot == 'CANON_NONCANON_COSIM': # inter-group
        ax.set_title(f"{title_mapping[canon_col]} ({col_or_grey}), $\\rho = {corr:.2f}$", fontsize = 17)
        
        # inter-group y axis is not mean cosine similarity, change this
        ylabel = 'Cosine Similarity'
        label_fontsize = 16

    elif cosim_to_plot == 'TOTAL_COSIM_MEAN': # total intra-group
        ax.set_title(f'Total data ({col_or_grey}), $\\rho = {corr:.2f}$', fontsize = 14)

        # add CI error band
        CI = sim_df['TOTAL_COSIM_CI']
        ci_lower = CI.apply(lambda x: x[0])
        ci_upper = CI.apply(lambda x: x[1])
        ax.fill_between(sim_df['START_year'], ci_lower, ci_upper, alpha=0.2, color = color)

        label_fontsize = 12

    elif cosim_to_plot == 'NONCANON_COSIM_MEAN': # non-canon intra-group
        ax.set_title(f'Total non-canon ({col_or_grey}), $\\rho = {corr:.2f}$', fontsize = 12)
        
        # add CI error band
        CI = sim_df['NONCANON_COSIM_CI']
        ci_lower = CI.apply(lambda x: x[0])
        ci_upper = CI.apply(lambda x: x[1])
        ax.fill_between(sim_df['START_year'], ci_lower, ci_upper, alpha=0.2, color = color)

        label_fontsize = 12

    else: # intra-group
        ax.set_title(f'{title_mapping[canon_col]} ({col_or_grey}), $\\rho = {corr:.2f}$', fontsize = 17)

        # add CI error band
        CI = sim_df['CANON_COSIM_CI']
        ci_lower = CI.apply(lambda x: x[0])
        ci_upper = CI.apply(lambda x: x[1])
        ax.fill_between(sim_df['START_year'], ci_lower, ci_upper, alpha=0.2, color = color)

        label_fontsize = 16

    # adjustments to plot layout
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    ax.set_xlabel('t', fontsize = label_fontsize)
    ax.set_ylabel(ylabel, fontsize = label_fontsize)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', which='major', length=4, width=1, labelsize=12)

def plot_grid(df:pd.DataFrame, color_subset:pd.DataFrame, canon_cols:list, w_size:int, cosim_to_plot:str, savefig:bool, filename:str):

    '''
    Plot grid of diachronic change plots for canon variables and color and greyscale data

    Parameters:
        - df: Dataframe with paintings data
        - color_subset: Dataframe with color paintings only
        - canon_cols: List of canon columns in dataframe
        - w_size: Rolling window size
        - cosim_to_plot: Cosine Similarity measure to plot
        - savefig: Whether to save the file or not
        - filename: Path to or name of output pdf file
    
    '''

    fig, axs = plt.subplots(2, 3, figsize=(19, 11))

    # plot a diachronic change lineplot for each canon variable in a grid, for greyscale and color
    for idx, col in enumerate(tqdm(canon_cols, desc=f"Plotting canon columns for {cosim_to_plot}")):

        # color
        plot_diachronic_change(w_size = 30, 
                            df = color_subset, 
                            canon_col = col, 
                            embedding_col = 'embedding', 
                            cosim_to_plot = cosim_to_plot, 
                            ax = axs[0, idx])
        
        # greyscale
        plot_diachronic_change(w_size = 30, 
                            df = df, 
                            canon_col = col, 
                            embedding_col = 'grey_embedding', 
                            cosim_to_plot = cosim_to_plot, 
                            ax = axs[1, idx])

        # remove y label for all columns except for first
        if idx != 0:
            axs[0, idx].set_ylabel('')
            axs[1, idx].set_ylabel('') 

    fig.tight_layout()

    if savefig:
            plt.savefig(filename, format='pdf', bbox_inches='tight')
    
def create_stacked_freqplot(df:pd.DataFrame, ax:plt.Axes, canon_col:str, year_col:str = 'start_year'):

    '''
    Create a stacked bar plot showing frequency counts of canon vs non-canon paintings per production year.

    Parameters:
        - df: DataFrame with SMK data
        - ax: Plt ax to plot on
        - canon_col: Canon variable to use
        - year_col: Name of column in df containing production years
    '''

    # change label names for plotting purposes
    column_mapping = {'canon': 'Canon', 'other': 'Non-canon'}

    bar_colors = {
        'Canon': '#08306B',      
        'Non-canon': '#6BAED6'  
    }

    # add title mapping for plotting purposes
    title_mapping = {'exb_canon': 'Exhibitions canon',
                     'smk_exhibitions': 'SMK exhibitions canon',
                     'on_display': 'On display canon'}

    # group by year
    groupobject = df.groupby([year_col, canon_col]).size().unstack()
    groupobject = groupobject.rename(columns=column_mapping)
    
    # create barplot
    groupobject.plot(kind='bar', stacked=True, ax=ax, edgecolor='none', width=0.95, color=[bar_colors.get(col, '#333333') for col in groupobject.columns])

    # Plot each 10th year label to not overcrowd the x-axis
    N = 10
    ax.set_xticks(range(0, len(groupobject), N))
    ax.set_xticklabels(groupobject.index[::N])

    ax.set_xlabel('Production year')
    ax.set_ylabel('Number of paintings')
    ax.set_title(f"{title_mapping[canon_col]}", pad=10)

    ax.legend(title='Canon status', loc='upper left', frameon=False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.tick_params(axis='both', which='major', length=4, width=1)

    ax.grid(True, linestyle='--', alpha=0.5)

    # Final layout cleanup
    ax.margins(x=0)

################################## SUPERVISED CLASSIFICATION ##################################
def run_classification(df:pd.DataFrame, embedding_col:str, canon_col:str, classifier:str, resample:bool, cv:bool, random_state:int):
    
    '''
    Run a supervised classification of canon/non-canon with optional sampling and cross validation steps.

    Parameters:
        - df: DataFrame with smk data
        - embedding_col: Name of column in df with embeddings
        - canon_col: Name of canon column variable to use
        - classifier: Whether to use 'mlp' or 'logistic' classifier
        - resample: Resample data or not
        - cv: Whether to cross validate or not
        - random_state: Set random state for reproducibility

    Returns:
        - Mean and SD cross-validation score if cv=True; otherwise None
    
    '''

    X = np.array(df[embedding_col].tolist())
    y = np.array(df[canon_col])

    if classifier == 'mlp':

        # specify mlp classifier parameters
        clf = MLPClassifier(random_state=random_state,
            hidden_layer_sizes=(100,),
            learning_rate='adaptive',
            early_stopping=False,
            verbose=False,
            max_iter=100)
    
    else:
        clf = LogisticRegression(random_state=random_state)

    if resample == True:
        # resample data
        if cv == True:

            # specify cross validation
            cv_methods = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

            # initiate pipeline from imbalanced-learn to apply sampling with cross validation
            imba_pipeline = make_pipeline(RandomUnderSampler(random_state=random_state), clf)
            scores = cross_val_score(imba_pipeline, X, y, scoring='f1_macro', cv=cv_methods)

            return f"Mean: {float(round(scores.mean(), 3))}, SD: {float(round(scores.std(), 3))}"
        
        else: # no cross validation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state, stratify=y)
            rus = RandomUnderSampler(random_state=random_state)

            # we only resample train data to avoid leakage to test data
            X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

            # fit classifier
            clf.fit(X_train_resampled, y_train_resampled)

            # predict on test data
            y_pred = clf.predict(X_test)
            print(classification_report(y_test, y_pred))
            return None
    
    else: # no resampling
        if cv == True:

            # specify cross validation & run classifier across splits
            cv_methods = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
            scores = cross_val_score(clf, X, y, scoring='f1_macro', cv=cv_methods)

            return f"Mean: {float(round(scores.mean(), 3))}, SD: {float(round(scores.std(), 3))}"
        
        else: # no resampling and no cross validation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state, stratify=y)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print(classification_report(y_test, y_pred))
            return None

def save_classification_results(canon_cols:list, models:list, sampling_methods:list, df:pd.DataFrame, embedding_col:str, col_or_grey:str, save_report:bool):

    '''
    Run classification models classifying canon vs non-canon and save/print results

    Parameters:
        - canon_cols: List of names of relevant columns with canon variables in dataframe
        - models: List of classification models to run classification with
        - sampling_methods: List of bools of whether to resample data or not
        - df: DataFrame with smk paintings data
        - embedding_col: Name of embedding column to use
        - col_or_grey: Whether color or greyscale data is used
        - save_report: whether to save the classification report or just print it
    
    '''

    # create table for printing results
    table = BeautifulTable()
    table.columns.header = ["unbalanced_logistic", "balanced_logistic", "unbalanced_mlp", "balanced_mlp"]
    table.rows.header = canon_cols

    # run classification for each canon column, specified model and sampling method
    for idx, col in enumerate(canon_cols):

        results_list = []
        
        for model in models:

            for sampling in sampling_methods:

                result = run_classification(df=df, 
                                            embedding_col = embedding_col, 
                                            canon_col = col, 
                                            classifier = model, 
                                            resample=sampling, 
                                            cv=True, 
                                            random_state=100)
                
                results_list.append(result)

        table.rows[idx] = results_list

    # print results and save to results folder
    print(f"CLASSIFICATION RESULTS, {col_or_grey} (MEAN STRATIFIED 10-FOLD CV MACRO F1 SCORES):")
    print(table)

    if save_report == True:
        # save classification report
        out_file = os.path.join('results', 'classification', f'{col_or_grey}_classification_report.txt')

        with open(out_file, 'w') as file:
                file.write(str(table))
                
################################## FUNCTIONS TO GATHER ALL OF THESE // PLOT EVERYTHING ##################################

def analysis_plots(df:pd.DataFrame, color_subset:pd.DataFrame, w_size:int, canon_cols:list):

    '''
    Gather analysis into single function to create and save plots

    Parameters:
        - df: DataFrame with smk data
        - color_subset: DataFrame with colored images only
        - w_size: Window size for rolling window analysis
        - canon_cols: List of canon variables (columns)
    '''
    # create and save PCA plots, greyscale and color embeddings
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    pca_binary(ax = axs[0], 
                df = color_subset, 
                embedding = 'embedding', 
                canon_category = 'exb_canon', 
                title = "Exhibition canon (color)")
    
    pca_binary(ax = axs[1], 
        df = df, 
        embedding = 'grey_embedding', 
        canon_category = 'exb_canon', 
        title = "Exhibition canon (greyscale)")
    
    plt.savefig(os.path.join('results', 'figs', 'PCA_exb_only.pdf'), format='pdf', dpi=300)

    # plot and save inter/intra group plots

    # canon
    plot_grid(df = df,
           color_subset=color_subset, 
            canon_cols = canon_cols,
            w_size= 30, 
            cosim_to_plot='CANON_COSIM_MEAN', 
            savefig=True,
            filename=os.path.join('results', 'figs', 'intra_canon_w30.pdf'))

    # intra, non-canon
    plot_grid(df = df, 
            color_subset = color_subset, 
            canon_cols = canon_cols,
            w_size= 30, 
            cosim_to_plot='NONCANON_COSIM_MEAN',
            savefig=True,
            filename=os.path.join('results', 'figs', 'intra_noncanon_w30.pdf'))
    
    # inter-group
    plot_grid(df = df, 
              color_subset = color_subset, 
              canon_cols = canon_cols,
              w_size= 30, 
              cosim_to_plot='CANON_NONCANON_COSIM', 
              savefig=True,
              filename=os.path.join('results', 'figs', 'inter_w30.pdf'))

    # plot and save total non-canon data intra-group plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    plot_diachronic_change(w_size = 30, 
                           df = color_subset, 
                           canon_col = 'total_canons', 
                           embedding_col = 'embedding', 
                           cosim_to_plot = 'NONCANON_COSIM_MEAN', 
                           ax = axs[0])
    
    plot_diachronic_change(w_size = 30, 
                           df = df, 
                           canon_col = 'total_canons', 
                           embedding_col = 'grey_embedding', 
                           cosim_to_plot = 'NONCANON_COSIM_MEAN', 
                           ax = axs[1])
    
    # remove y label from second plot
    axs[1].set_ylabel("")

    plt.savefig(os.path.join('results', 'figs', 'total_noncanon_w30.pdf'), format='pdf', dpi=300)

################################## DATASET VISUALIZATIONS ##################################
def pca_icons(ax:plt.Axes, df:pd.DataFrame, ds: Dataset, embedding:str, image_col:str, filename:str=None):

    '''
    Plot PCA of image embeddings with painting icons instead of points

    Parameters:
        - ax: Plt axes to plot on
        - df: Pandas dataframe with SMK data
        - ds: HuggingFace dataset with images
        - embedding: Name of embedding column in df
        - image_col: Name of image column in ds
        - filename: Filename to save figure under
    '''
    # convert list of embeddings to array
    embeddings_array = np.array(df[embedding].to_list(), dtype=np.float32)

    # to 2 dimensions
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(embeddings_array)
    df_pca = pd.DataFrame(pca_results, columns=["PCA1", "PCA2"])
    
    ax.scatter(df_pca["PCA1"], df_pca["PCA2"], color='white')

    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")

    ax.axis("equal")
    ax.set_axis_off()

    # instead of points on scatterplot, plot paintings instead
    if image_col == 'grey_image':

        import cv2 

        grey_images = []

        feature = Image_ds(decode=False)

        for i in tqdm(range(len(ds))):
            image = ds[i]['image']
            image_greyscale = image.convert('L')
            image_encoded = feature.encode_example(image_greyscale) # in order to add an image to a HF dataset column, the image needs to be encoded properly
            grey_images.append(image_encoded)

        ds = ds.add_column('grey_image', grey_images)
        ds = ds.cast_column('grey_image', Image_ds(decode=True))

        # add encoded column to dataframe as well

        df['grey_image'] = grey_images

        def getImage(img):
            return OffsetImage(np.array(img), zoom=.02, cmap='gray') # need to change color map if plotting greyscale; matplotlib defaults 1-channel images to different cmap otherwise..
    
    else:
        def getImage(img):
            return OffsetImage(np.array(img), zoom=.02)

    for index, row in df_pca.iterrows():
        # add images to plot
        ab = AnnotationBbox(getImage(ds[index][image_col]), (row["PCA1"], row["PCA2"]), frameon=False)
        ax.add_artist(ab)

    if filename:
        plt.savefig(os.path.join('results', 'figs', filename), format='eps', dpi=1200)

    np.seterr(divide='ignore', invalid='ignore')

def umap_plot(ax:plt.Axes, df:pd.DataFrame, ds:Dataset, embedding:str, n_components:int = 50):

    '''
    Plot all paintings with UMAP
    
    Parameters:
        - ax: Plt axes to plot on
        - df: Dataframe with embeddings
        - ds: Dataset with color images
        - embedding: name of embedding col in df
        - filename: Name to save the plot as
        - n_components: Number of PCA components for initial dimensionality reduction
    '''
    def getImage(img):
        return OffsetImage(np.array(img), zoom=.02)

    embeddings_array = np.array(df[embedding].to_list(), dtype=np.float32)

    # reduce dimensionality
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(embeddings_array)

    # fit UMAP on PCA-reduced data
    X = np.array(pca_results)
    umap_fitted = UMAP(n_components=2, random_state=42).fit_transform(X)
    df_umap = pd.DataFrame(umap_fitted, columns=["umap1", "umap2"])
    
    ax.scatter(df_umap['umap1'], df_umap['umap2'], color='white') # make invisible

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.axis("equal")
    ax.set_axis_off()

    for index, row in df_umap.iterrows():
        # add images to plot
        ab = AnnotationBbox(getImage(ds[index]['image']), (row["umap1"], row["umap2"]), frameon=False)
        ax.add_artist(ab)

    plt.savefig(os.path.join('results', 'figs', f'umap_n{n_components}.eps'), format='eps', dpi=1200)

    np.seterr(divide='ignore', invalid='ignore')

def plot_pca_comparison(df: pd.DataFrame,
                        ds: Dataset,
                        start_period: tuple = (1750, 1780),
                        end_period: tuple = (1781, 1810),
                        embedding_type: str = 'embedding',
                        canon_filter: str = 'all',
                        title: str = None,
                        filename: str = None):
    
    '''
    Generate PCA plots for two time periods.

    Parameters: 
        - df: Dataframe with SMK data
        - ds: HuggingFace dataset with images
        - start_period: First period to plot PCA for
        - end_period: Second period to plot PCA for
        - embedding_type: Name of embedding col
        - canon_filter: Filter value for 'exb_canon' column ('all', 'canon' or 'other')
        - title: Title on plot
        - filename: Name of output file

    '''

    # filter data based on chosen periods and canon_filter
    def filter_data(dataframe, year_range, canon):
        subset = dataframe.query(f'start_year >= {year_range[0]} and start_year <= {year_range[1]}')
        if canon != 'all':
            subset = subset.query(f'exb_canon == "{canon}"')
        idxs = subset.index.tolist()
        return subset, ds.select(idxs)

    df1, ds1 = filter_data(df, start_period, canon_filter)
    df2, ds2 = filter_data(df, end_period, canon_filter)

    # plot the two PCA's next to each other
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    pca_icons(axs[0], df1, ds1, embedding_type, 'image', None)
    pca_icons(axs[1], df2, ds2, embedding_type, 'image', None)

    if title:
        fig.suptitle(title, fontsize=15)

    if filename:
        plt.savefig(os.path.join('results', 'figs', filename), bbox_inches='tight', format='pdf', dpi=1200)

def dataset_visualizations(canon_cols:list, df:pd.DataFrame, ds:Dataset, color_subset:pd.DataFrame, ds_color:Dataset):
    
    '''
    Gather all functions for creating visualizations of the paintings dataset

    Parameters: 
        - canon_cols: List of names of relevant columns with canon variables in dataframe
        - df: DataFrame with paintings data
        - ds: HuggingFace dataset with images
        - color_subset: Subset of dataframe with color-only images
        - ds_color: Subset of dataset with color-only images

    '''
    # plot and save canon/painting frequency
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    for idx, col in enumerate(canon_cols):
       create_stacked_freqplot(df = df, 
                               ax = axs[idx], 
                               canon_col = col, 
                               year_col = 'start_year')
       if idx != 0:
           axs[idx].set_ylabel('')   # Remove Y label for all columns except for first
           axs[idx].legend().remove()
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'figs', 'canon_frequency.pdf'), format='pdf', dpi=300)
    
    # plot and save PCAs with painting icons 

    fig, axs = plt.subplots(1, 1, figsize=(30, 20))

    print('Creating PCA plots....')

    pca_icons(axs, 
              color_subset, 
              ds_color, 
              'embedding', 
              'image', 
              'pca_paitings_color.eps') # saving to .eps ensures good quality but creates really large files, so output files are added to .ignore

    pca_icons(axs, 
              df, 
              ds, 
              'grey_embedding', 
              'grey_image',  
              'pca_paitings_grey.eps')

    plot_pca_comparison(color_subset,
                        ds_color, 
                        canon_filter='all', 
                        filename='pca_innovation_total.pdf')

    plot_pca_comparison(color_subset,
                        ds_color, 
                        canon_filter='canon', 
                        filename='pca_innovation_canon.pdf')
    
    plot_pca_comparison(color_subset,
                        ds_color, 
                        canon_filter='other', 
                        filename='pca_innovation_non_canon.pdf')
    
    # plot and save color UMAP
    fig, axs = plt.subplots(1, 1, figsize=(20, 15))
    print('Creating UMAP plot...')
    umap_plot(axs, color_subset, ds_color, 'embedding')


################################## IMAGE DOWNLOADING ##################################

def download_image(url:str):
    '''
    Download image from SMK thumbnail URL and encode it to a HuggingFace Image feature

    Parameters:
        - url: URL to image
    
    Returns:
        Encoded image / pd.NA if download fails
    '''
    # create image feature to use for encoding
    feature = Image_ds(decode=False) 

    try:
        img = Image.open(requests.get(url, stream=True).raw) # stream=True enables to download image in chunks and not all at once

        # encode the PIL images to image feature
        image_encoded = feature.encode_example(img)

        return image_encoded

    except Exception as e:
        print(f"Error processing image: {e}")
        return pd.NA

def add_image_col(ds: Dataset, url_col:str):
    '''
    Download all images from thumbnails in a column and cast them to PIL format for HuggingFace dataset

    Parameters:
        - ds: HuggingFace dataset object containing column with image URLs
        - url_col: name of column with Image URLs
    
    Returns:
        - Dataset with decoded image column
    '''
    # download images from URLs
    images = [download_image(url) for url in tqdm(ds[url_col], desc= 'Downloading images')]
    
    # add column to dataset
    ds = ds.add_column('image', images)

    # decode back to PIL format
    ds = ds.cast_column('image', Image_ds(decode=True))

    return ds