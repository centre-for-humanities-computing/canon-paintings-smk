from PIL import Image
import os
import pandas as pd
import numpy as np
import datasets
from tqdm import tqdm
from datasets import Image as Image_ds # change name because of similar PIL module
from datasets import Dataset
from datasets import load_dataset
import urllib.parse
import json
import pickle
import requests
import matplotlib.pyplot as plt 
from sklearn.neighbors import NearestNeighbors
import re
import sys
sys.path.append(os.path.abspath(".."))
#from src.utils import plot_neighbors, pca_binary, plot_pca_scale, plot_dendrogram
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
from random import sample
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib.patches import Patch
from sklearn.metrics.pairwise import cosine_distances
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.utils import resample
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate, cross_val_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from collections import Counter
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from beautifultable import BeautifulTable

#sys.path.append(os.path.abspath(".."))
from utils import WindowedRollingDistance
from utils import calc_vector_histogram
from scipy.ndimage import gaussian_filter1d

# we get a lot of annoying warnings from sklearn so we suppress them
import warnings
warnings.filterwarnings('ignore')

##################### PCA ##################################

def pca_binary(ax, df, embedding, canon_category, title):
    embeddings_array = np.array(df[embedding].to_list(), dtype=np.float32)
    
    color_mapping = {'other': '#129525', 'canon': '#75BCC6'}

    # to 2 dimensions
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(embeddings_array)
    df_pca = pd.DataFrame(pca_results, columns=["PCA1", "PCA2"])
    df_pca["canon"] = df[canon_category].values

    # Plot each category
    for category in df_pca["canon"].unique():
        subset = df_pca[df_pca["canon"] == category]

        #marker = markers_dict.get(category) 
        #alpha = alpha_dict.get(category)
        
        ax.scatter(
            subset["PCA1"],
            subset["PCA2"],
            color=color_mapping.get(category),
            label=category,
            alpha=0.4,
            edgecolor='black',
            s=110,
            marker='o' #marker
        )

    ax.set_title(title)
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")

    legend_handles = [Patch(facecolor=color, label=label) for label, color in color_mapping.items()]
    ax.legend(handles=legend_handles, loc='upper right')

    ax.axis("equal")

    # supress warnings 

    np.seterr(divide='ignore', invalid='ignore')


##################### DIACHRONIC CHANGE ##################################

def get_cosim_mean_std(groups_dict, embedding_col, key):

    '''
    Calculate mean and SD cosine similarity for embedding column of dataset in groups_dict
    '''

    data = groups_dict[key]
    embeddings = np.stack(data[embedding_col].values)
    mean_cosim = cosine_similarity(embeddings).mean()
    std_cosim = cosine_similarity(embeddings).std()
    
    return mean_cosim, std_cosim

def create_groups(df, year_col, canon_col, year_range):

    '''
    Create dict with groups of canon, non-canon and total paintings in specified time window
    '''
    
    canon = df.loc[(df[year_col].isin(year_range)) & (df[canon_col] == 'canon')]
    df_total = df.loc[df[year_col].isin(year_range)]
    non_canon = df.loc[(df[year_col].isin(year_range)) & (df[canon_col] == 'other')]

     # create dict
    groups = {'canon': canon, 'df_total': df_total, 'non_canon': non_canon}

    return groups

def get_all_cosims(df, year_col, canon_col, year_range, embedding_col, sampling, sample_size, run):

    '''
    Calculate all cosine similarity measures for current time window and return as dict. 

    First creates groups of canon, non-canon and total paintings in the current time window.
    Next, it calculates the mean cosine similarity of the embeddings in the current window for each group.
    The cosine similarity between the mean canon and non-canon embeddings is also calculated.
    '''

    groups = create_groups(df, year_col, canon_col, year_range)

    if sampling == True:
        for key in groups:
            group = groups[key]
            groups[key] = group.sample(sample_size, random_state=run) if len(group) > sample_size else group

    # get the mean embeddings of the current window for each group
    canon_mean = groups['canon'][embedding_col].mean(axis=0)
    non_canon_mean = groups['non_canon'][embedding_col].mean(axis=0)

    temp = {} 

    # get the mean cosine similarity between mean canon embedding and mean non-canon embedding for this time window
    
    #canon_noncanon_similarity = cosine_similarity(np.stack([non_canon_mean, canon_mean])).mean()
    canon_noncanon_similarity = cosine_similarity(np.stack([non_canon_mean, canon_mean]))[0][1]
    temp['CANON_NONCANON_COSIM'] = canon_noncanon_similarity

    # get mean cosine similarity of canon embeddings for this time window
    canon_mean, canon_std = get_cosim_mean_std(groups, embedding_col, 'canon')
    temp['CANON_COSIM_MEAN'] = canon_mean 
    temp['CANON_COSIM_STD'] = canon_std

    # get mean cosine similarity of non-canon embeddings for this time window
    nc_mean, nc_std = get_cosim_mean_std(groups, embedding_col, 'non_canon')
    temp['NONCANON_COSIM_MEAN'] = nc_mean
    temp['NONCANON_COSIM_STD'] = nc_std

    # get mean cosine similarity of all data for this time window
    t_mean, t_std = get_cosim_mean_std(groups, embedding_col, 'df_total')
    temp['TOTAL_COSIM_MEAN'] = t_mean
    temp['TOTAL_COSIM_STD'] = t_std

    temp['n_paintings'] = [len(groups['df_total']), len(groups['canon']), len(groups['non_canon'])]
    
    return temp

def run_change_analysis(w_size, df, canon_col, embedding_col, step_size=1, year_col='start_year', n_runs=1, sampling=False, sample_size=0, simulate=False, num_simulations=0, sim_type='none'):
    # raise error if w size is smaller than 5
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

            if simulate == True:
                temp = simulate_all_cosims(df, year_col, canon_col, year_range, embedding_col, sampling, sample_size, run, num_simulations, sim_type)

            else:
                temp = get_all_cosims(df, year_col, canon_col, year_range, embedding_col, sampling, sample_size, run)
            
            # add cosine similarity measure to dict for dict at time window
            mean_similarity_dict[range_label] = temp
    
    # create dataframe from dict
    sim_df = pd.DataFrame.from_dict(mean_similarity_dict, orient='index').reset_index()
    sim_df = sim_df.rename(columns={"index": "year_RANGE"})

    # add start year column
    sim_df['START_year'] = sim_df['year_RANGE'].apply(lambda x: int(x.split('-')[0]))

    # make sure there's at least 2 paintings in each group

    sim_df['n_paintings']

    part_canon_list = []

    for l in sim_df['n_paintings']:
        total = l[0]
        canon = l[1]

        part_canon = canon / total
        part_canon_list.append(part_canon)


    #print(f"mean canon part: {np.mean(part_canon_list)}, min canon part: {np.min(part_canon_list)}, max canon part: {np.max(part_canon_list)}")
    #print(sim_df['n_paintings'].apply(lambda x: min(x)).min(), "is the smallest group size in a window")

    return sim_df

def plot_diachronic_change(w_size, df, canon_col, embedding_col, cosim_to_plot, ax, step_size=1, year_col='start_year', n_runs=1, sampling=False, sample_size=0, simulate=False, num_simulations=0, sim_type='none', cutoff=5, color='#75BCC6'):

    plt.style.use('default')

    # get dataframe of cosine similarity for each time window for chosen canon measure 
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
                simulate=simulate,
                num_simulations=num_simulations,
                sim_type=sim_type
    )

    min_group_idx = int(sim_df['n_paintings'].explode().idxmin())
    min_group_size = min(sim_df['n_paintings'].iloc[min_group_idx])
    #min_group_year = sim_df['START_year'].iloc[min_group_idx]
    #min_group_cosim = sim_df[cosim_to_plot].iloc[min_group_idx]

    # remove first n and last n windows
    #sim_df = sim_df.iloc[cutoff:-cutoff]
    
        # get correlation between year and cosine similarity
    corr, pval = spearmanr(sim_df['START_year'], sim_df[cosim_to_plot])

    # plot change over time
    ax.plot(sim_df['START_year'], sim_df[cosim_to_plot], color=color, linewidth=2, alpha=0.7)
    #ax.plot(min_group_year, min_group_cosim, marker='x', color='red', markersize=5, label='Minimum group size')

    if pval < 0.01:
        greater_dir = '<'
    else:
        greater_dir = '>'

    col_or_grey = 'colored' if embedding_col == 'embedding' else 'greyscaled'

    ylabel = 'Mean Cosine Similarity'

    if cosim_to_plot == 'CANON_NONCANON_COSIM':
        ax.set_title(f'{canon_col} vs non-canon, {col_or_grey}, small. grp. = {min_group_size}')
        ylabel = 'Cosine Similarity'

    elif cosim_to_plot == 'TOTAL_COSIM_MEAN':
        ax.set_title(f'Total data, {col_or_grey}, small. grp. = {min_group_size}')

    else:
        ax.set_title(f'{canon_col}, {col_or_grey}, small. grp. = {min_group_size}')

    # create plot
    ax.set_xlabel(year_col)
    ax.set_ylabel(ylabel)
    #ax.legend(fontsize=8)

    #ax.set_xlim(sim_df['START_year'].iloc[0], sim_df['START_year'].iloc[-1])
    #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper right in axes coords
    ax.text(0.5, 0.95, f"r = {round(corr, 2)}, p{greater_dir}0.1", transform=ax.transAxes, fontsize=10,
        verticalalignment='top')
    
def create_stacked_freqplot(df, ax, canon_col, w_size, year_col = 'start_year'):

    groupobject = df.groupby([year_col, canon_col]).size().unstack()
    groupobject.plot(kind='bar', stacked=True, ax=ax)

    N = w_size
    ax.set_xticks(range(0, len(groupobject), N))
    ax.set_xticklabels(groupobject.index[::N])

    ax.set_title(f'{canon_col} frequency')
    ax.set_xlabel(year_col)   
    ax.set_ylabel('Number of paintings')

    #ax.legend(title='Canon status', loc='upper right')

def plot_grid(df, color_subset, canon_cols, w_size, cosim_to_plot, title, savefig, filename):

    fig, axs = plt.subplots(3, len(canon_cols), figsize=(len(canon_cols)*6, 15))

    for idx, col in enumerate(canon_cols):

        plot_diachronic_change(w_size = 30, 
                            df = color_subset, 
                            canon_col = col, 
                            embedding_col = 'embedding', 
                            cosim_to_plot = cosim_to_plot, 
                            ax = axs[0, idx])
        
        plot_diachronic_change(w_size = 30, 
                            df = df, 
                            canon_col = col, 
                            embedding_col = 'grey_embedding', 
                            cosim_to_plot = cosim_to_plot, 
                            ax = axs[1, idx])

        create_stacked_freqplot(df=df, w_size = 30, ax = axs[2, idx], canon_col=col)


    fig.suptitle(f'{title}, w_size = {w_size}', size = 20, y=0.95)

    if savefig==True:
        plt.savefig(filename, format='png', dpi=1200)
    
##################### ENTROPY / CHANGE DETECTION ##################################

def calc_signals(df, w_size, embedding_col, year_col):
    '''
    Calculate novelty, transience and resonance signals from image embeddings
    '''
    # convert embedding to array
    X = np.array(df[embedding_col])

    # convert array to probability distributions and back to array
    X_prob = [calc_vector_histogram(vect, bins=256) for vect in X]
    X_prob = np.array(X_prob)

    # calculate signal
    w_main = WindowedRollingDistance(
        measure='jensenshannon',
        window_size=w_size,
        estimate_error=True)

    signal = w_main.fit_transform(X_prob)

    # convert to dataframe
    signal_df = pd.DataFrame(signal)

    # add information about production year
    signal_df['year'] = df[year_col]

    return signal_df

def plot_canon_novelty(df, canon_noncanon, canon_col, embedding_col, w_size, year_col, col_or_grey, ax):
    
    # define canon df based on chosen canon variable

    if canon_noncanon == 'total':
        df_canon = df

    elif canon_noncanon == 'non_canon':
        df_canon = df[df[canon_col] == 'other']

    else:
        df_canon = df[df[canon_col] == 'canon']

    # sort by start year, lowest to highest
    df_canon_sorted = df_canon.sort_values(by=year_col).reset_index(drop=True)

    # calculate novelty and resonance signals (windowed rolling distance function
    signal_df = calc_signals(df_canon_sorted, w_size, embedding_col, year_col)

    # exclude the first w_size rows as they are empty
    n_df = signal_df.iloc[w_size:, :]
    n_hat = n_df['N_hat'].tolist() # get novelty signal
    n_years = n_df['year'].tolist() # get year

    # plot raw novelty signal as well as smoothed signal with a gaussian filter
    ax.plot(n_years, n_hat, alpha=0.5)
    ax.plot(n_years, gaussian_filter1d(n_hat, 5), label='Novelty (Gaussian filter, sigma = 5)', color='blue')
    ax.set_ylabel(f'Average windowed Jensen-Shannon Distance (w={w_size})')
    ax.set_xlabel(year_col)
    ax.set_title(f'{canon_col}, {col_or_grey}, w_size = {w_size}')
    ax.legend(loc='upper right')

##################### SUPERVISED CLASSIFICATION ##################################
def run_classification(df, embedding_col, canon_col, classifier, resample, sample_before, cv, random_state):
    
    X = np.array(df[embedding_col].tolist())
    y = np.array(df[canon_col])

    if classifier == 'mlp':
        clf = MLPClassifier(random_state=random_state,
            hidden_layer_sizes=(100,),
            learning_rate='adaptive',
            early_stopping=False,
            verbose=False,
            max_iter=100)
    
    else:
        clf = LogisticRegression(random_state=random_state)

    if resample == True:
        if sample_before == True:
            rus = RandomUnderSampler(random_state=random_state)
            X, y = rus.fit_resample(X, y)
            print(sorted(Counter(y).items()))

            if cv == True: 
                scores = cross_val_score(clf, X, y, cv=10, scoring='f1_macro')
                #print(f"Mean macro f1 cross val score: {scores.mean()}, sd: {scores.std()}")
                return round(scores.mean(), 3)
            
            else: # if resample == True and sample_before == True, but no cv
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state, stratify=y)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                print(classification_report(y_test, y_pred))
                return 

        else: # resampling, but sample after splitting
            if cv == True:
                imba_pipeline = make_pipeline(RandomUnderSampler(random_state=random_state), clf)
        
                cross_vals = cross_val_score(imba_pipeline, X, y, scoring='f1_macro', cv=10)

                #print(f"Mean macro f1 cross val score: {cross_vals.mean()}, sd: {cross_vals.std()}")
                return round(cross_vals.mean(), 3)
            
            else: # no cross validation
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state, stratify=y)
                rus = RandomUnderSampler(random_state=random_state)
                X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
                clf.fit(X_train_resampled, y_train_resampled)
                y_pred = clf.predict(X_test)
                print(classification_report(y_test, y_pred))
                return 
    
    else: # no resampling
        if cv == True:
            scores = cross_val_score(clf, X, y, cv=10, scoring='f1_macro')
            #print(f"Mean macro f1 cross val score: {scores.mean()}, sd: {scores.std()}")
            return round(scores.mean(), 3)
        
        else: # no resampling and no cross validation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state, stratify=y) #?
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print(classification_report(y_test, y_pred))
            return


##################### FUNCTIONS TO GATHER ALL OF THESE // PLOT EVERYTHING ##################################
def all_analyses_plots(df, color_subset, canon_cols, plot_suffix, plot_folder, inter_intra_w, novelty_w):

    # PCA plots
    fig, axs = plt.subplots(2, len(canon_cols), figsize=(15, 10))

    for idx, col in enumerate(canon_cols):

        pca_binary(ax = axs[0,idx], 
                   df = color_subset, 
                   embedding = 'embedding', 
                   canon_category = col, 
                   title = f"{col}, colored")
        
        pca_binary(ax = axs[1, idx], 
            df = df, 
            embedding = 'grey_embedding', 
            canon_category = col, 
            title = f"{col}, greyscale")

    fig.suptitle('PCA by canon category', size = 15, y=0.95)
    plt.savefig(os.path.join(plot_folder, f'PCA_{plot_suffix}.png'), format='png', dpi=1200)

    # intra-group, canon, non-canon and total data

    # total data
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    plot_diachronic_change(w_size = inter_intra_w, 
                df = color_subset, 
                canon_col = 'exb_canon', 
                embedding_col = 'embedding', 
                cosim_to_plot = 'TOTAL_COSIM_MEAN', 
                ax = axs[0])

    plot_diachronic_change(w_size = inter_intra_w, 
            df = df, 
            canon_col = 'exb_canon', 
            embedding_col = 'grey_embedding', 
            cosim_to_plot = 'TOTAL_COSIM_MEAN', 
            ax = axs[1])
    
    fig.suptitle(f'Intra-group analysis for total data, w_size = {inter_intra_w}', size = 15, y=0.97)
    plt.savefig(os.path.join(plot_folder, f'intra_total_w{inter_intra_w}_{plot_suffix}.png'), format='png', dpi=1200)

    # intra, canon

    plot_grid(df = df,
               color_subset=color_subset, 
               canon_cols = canon_cols,
                w_size= inter_intra_w, 
                cosim_to_plot='CANON_COSIM_MEAN', 
                title='Intra-group analysis for all canon variables', 
                savefig=True,
                filename=os.path.join(plot_folder, f'intra_canon_w{inter_intra_w}_{plot_suffix}.png'))
    
    # intra, non-canon
    plot_grid(df = df, 
              color_subset = color_subset, 
              canon_cols = canon_cols,
            w_size= inter_intra_w, 
            cosim_to_plot='NONCANON_COSIM_MEAN', 
            title='Intra-group analysis for all non-canon',
            savefig=True,
            filename=os.path.join(plot_folder, f'intra_noncanon_w{inter_intra_w}_{plot_suffix}.png'))

    # inter-group
    plot_grid(df = df, 
              color_subset = color_subset, 
              canon_cols = canon_cols,
                w_size= inter_intra_w, 
                cosim_to_plot='CANON_NONCANON_COSIM', 
                title='Inter-group analysis for all canon variables',
                savefig=True,
                filename=os.path.join(plot_folder, f'inter_w{inter_intra_w}_{plot_suffix}.png'))

    # NOVELTY // ENTROPY PLOTS

    # total data
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    plot_canon_novelty(df, 'total', 'Total data', 'embedding', novelty_w, 'start_year', 'greyscale', axs[0])
    plot_canon_novelty(color_subset, 'total', 'Total data', 'embedding', novelty_w, 'start_year', 'colored', axs[1])
    fig.suptitle('Novelty signal for total data with mean embedding per year (all data)', size = 13, y=0.97)
    plt.savefig(os.path.join(plot_folder, f'novelty_w=mean_embed_year_w{novelty_w}_{plot_suffix}.png'), format='png', dpi=1200)

    # canon data
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    for idx, col in enumerate(canon_cols):
        plot_canon_novelty(df = color_subset, 
                           canon_noncanon = 'canon',
                            canon_col = col, 
                            embedding_col = 'embedding', 
                            w_size = novelty_w, 
                            year_col = 'start_year', 
                            col_or_grey = 'colored', 
                            ax = axs[0, idx])

        plot_canon_novelty(df = df,
                           canon_noncanon = 'canon',
                            canon_col = col,
                            embedding_col = 'grey_embedding', 
                            w_size = novelty_w,
                            year_col = 'start_year', 
                            col_or_grey= 'greyscale', 
                            ax = axs[1, idx])

    fig.suptitle(f'Novelty signal for all canon variables, w_size = {novelty_w}', size = 20, y=0.97)
    plt.savefig(os.path.join(plot_folder, f'novelty_w=mean_embed_year_canon_w{novelty_w}_{plot_suffix}.png'), format='png', dpi=1200)
    
    # non-canon data

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for idx, col in enumerate(canon_cols):
        plot_canon_novelty(df = color_subset, 
                           canon_noncanon = 'non_canon',
                            canon_col = col, 
                            embedding_col = 'embedding', 
                            w_size = novelty_w, 
                            year_col = 'start_year', 
                            col_or_grey = 'colored', 
                            ax = axs[0, idx])

        plot_canon_novelty(df = df,
                           canon_noncanon = 'non_canon',
                            canon_col = col,
                            embedding_col = 'grey_embedding', 
                            w_size = novelty_w,
                            year_col = 'start_year', 
                            col_or_grey= 'greyscale', 
                            ax = axs[1, idx])

    fig.suptitle(f'Novelty signal for all non-canon, w_size = {novelty_w}', size = 20, y=0.97)
    plt.savefig(os.path.join(plot_folder, f'novelty_w=mean_embed_year_noncanon_w{novelty_w}_{plot_suffix}.png'), format='png', dpi=1200)

def print_classification_results(canon_cols, models, sampling_methods, df, embedding_col, col_or_grey, report_suffix, out_folder):

    table = BeautifulTable()
    table.columns.header = ["unbalanced_logistic", "balanced_logistic", "unbalanced_mlp", "balanced_mlp"]
    table.rows.header = canon_cols

    for idx, col in enumerate(canon_cols):

        results_list = []
        
        for model in models:

            for sampling in sampling_methods:

                result = run_classification(df=df, 
                                            embedding_col = embedding_col, 
                                            canon_col = col, 
                                            classifier = model, 
                                            resample=sampling, 
                                            sample_before=False, 
                                            cv=True, 
                                            random_state=100)
                
                results_list.append(result)

        table.rows[idx] = results_list

    print(f"CLASSIFICATION RESULTS, {col_or_grey} (MEAN 10-FOLD CV MACRO F1 SCORES):")
    print(table)

    # save classification report
    out_file = os.path.join(out_folder, f'{col_or_grey}_classification_report_{report_suffix}.txt')

    with open(out_file, 'w') as file:
                file.write(str(table))
        

