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
#sys.path.append(os.path.abspath(".."))
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
from collections import Counter
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from beautifultable import BeautifulTable
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from umap import UMAP
import cv2 
from PIL import Image

#sys.path.append(os.path.abspath(".."))
from src.utils import WindowedRollingDistance
from src.utils import calc_vector_histogram
from scipy.ndimage import gaussian_filter1d

# we get a lot of annoying warnings from sklearn so we suppress them
import warnings
warnings.filterwarnings('ignore')

# default params
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

##################### PCA ##################################

def pca_binary(ax, df, embedding, canon_category, title):
    
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

        #marker = markers_dict.get(category) 
        #alpha = alpha_dict.get(category)
        
        ax.scatter(
            subset["PCA1"],
            subset["PCA2"],
            color=color_mapping.get(category),
            label=label_mapping.get(category),
            alpha=0.6,
            edgecolor='black',
            s=110,
            marker='o' #marker
        )

    #for spine in ax.spines.values():
        #spine.set_linewidth(1.5)

    #for spine in ax.spines.values():
        #spine.set_visible(False)

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


##################### DIACHRONIC CHANGE ##################################

def calc_bootstrap_CI_intra(n_bootstraps, similarities):
    rng = np.random.default_rng(seed=42)
    means = []

    for _ in range(n_bootstraps):
        sample = rng.choice(similarities, size=len(similarities), replace=True)
        means.append(np.mean(sample))
        
    # Confidence interval (e.g., 95%)
    lower = np.percentile(means, 2.5)
    upper = np.percentile(means, 97.5)
    #print(f"Mean: {np.mean(upper_triangle):.4f}, 95% CI: [{lower:.4f}, {upper:.4f}]")

    CI = [lower, upper]

    return CI 

def calc_bootstrap_CI_inter(groups, embedding_col, n_bootstrap):

    sims = []

    canon_embeddings = np.stack(groups['canon'][embedding_col].values)
    noncanon_embeddings = np.stack(groups['non_canon'][embedding_col].values)

    for _ in range(n_bootstrap):

        idx_canon = np.random.choice(len(canon_embeddings), size=len(canon_embeddings), replace=True)
        idx_noncanon = np.random.choice(len(noncanon_embeddings), size=len(noncanon_embeddings), replace=True)

        sample_canon = canon_embeddings[idx_canon]
        sample_noncanon = noncanon_embeddings[idx_noncanon]
        
        mean_canon = sample_canon.mean(axis=0)
        mean_noncanon = sample_noncanon.mean(axis=0)
        
        sim = cosine_similarity(np.stack([mean_noncanon, mean_canon]))[0][1]
        sims.append(sim)
    
    lower = np.percentile(sims, (100 - 0.95) / 2)
    upper = np.percentile(sims, 100 - (100 - 0.95) / 2)

    CI = [lower, upper]

    mean_sim = np.mean(sims)
    
    return mean_sim, CI

def get_cosim_mean_std(groups_dict, embedding_col, key):

    '''
    Calculate mean and SD cosine similarity for embedding column of dataset in groups_dict
    '''

    data = groups_dict[key]
    embeddings = np.stack(data[embedding_col].values)
    #mean_cosim = cosine_similarity(embeddings).mean()

    similarities = cosine_similarity(embeddings)
    
    # Remove diagonal (self-similarities = 1.0)
    #upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]

    mean_cosim = similarities.mean()
    #mean_cosim = upper_triangle.mean()

    CI = calc_bootstrap_CI_intra(1000, similarities)
    
    return mean_cosim, CI

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

    _, inter_CI = calc_bootstrap_CI_inter(groups, embedding_col, 1000)

    temp = {} 

    # get the mean cosine similarity between mean canon embedding and mean non-canon embedding for this time window
    
    #canon_noncanon_similarity = cosine_similarity(np.stack([non_canon_mean, canon_mean])).mean()
    canon_noncanon_similarity = cosine_similarity(np.stack([non_canon_mean, canon_mean]))[0][1]
    temp['CANON_NONCANON_COSIM'] = canon_noncanon_similarity
    temp['CANON_NONCANON_COSIM_CI'] = inter_CI

    # get mean cosine similarity of canon embeddings for this time window
    canon_mean, canon_CI = get_cosim_mean_std(groups, embedding_col, 'canon')
    temp['CANON_COSIM_MEAN'] = canon_mean 
    temp['CANON_COSIM_CI'] = canon_CI

    # get mean cosine similarity of non-canon embeddings for this time window
    nc_mean, nc_CI = get_cosim_mean_std(groups, embedding_col, 'non_canon')
    temp['NONCANON_COSIM_MEAN'] = nc_mean
    temp['NONCANON_COSIM_CI'] = nc_CI

    # get mean cosine similarity of all data for this time window
    t_mean, t_CI = get_cosim_mean_std(groups, embedding_col, 'df_total')
    temp['TOTAL_COSIM_MEAN'] = t_mean
    temp['TOTAL_COSIM_CI'] = t_CI

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

def plot_diachronic_change(w_size, df, canon_col, embedding_col, cosim_to_plot, ax, step_size=1, year_col='start_year', n_runs=1, sampling=False, sample_size=0, simulate=False, num_simulations=0, sim_type='none', cutoff=5, color='C0'):

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

    #print(sim_df.columns)
    min_group_idx = int(sim_df['n_paintings'].explode().idxmin())
    min_group_size = min(sim_df['n_paintings'].iloc[min_group_idx])
    
    # get correlation between year and cosine similarity
    corr, pval = spearmanr(sim_df['START_year'], sim_df[cosim_to_plot])

    if embedding_col == 'grey_embedding':
        color = 'grey'

    # plot change over time
    ax.plot(sim_df['START_year'], 
            sim_df[cosim_to_plot], 
            color=color, 
            linewidth=3, 
            alpha=1)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    if pval < 0.01:
        greater_dir = '<'
    else:
        greater_dir = '>'

    col_or_grey = 'colored' if embedding_col == 'embedding' else 'greyscaled'

    title_mapping = {'exb_canon': 'Exhibitions canon',
                     'smk_exhibitions': 'SMK exhibitions canon',
                     'on_display': 'On display canon'}

    ylabel = 'Mean Cosine Similarity'

    #print(sim_df['CANON_COSIM_STD'])

    if cosim_to_plot == 'CANON_NONCANON_COSIM':
        ax.set_title(f"{title_mapping[canon_col]} ({col_or_grey}), $\\rho = {corr:.2f}$", fontsize = 17)
        ylabel = 'Cosine Similarity'

        # add confidence interval band
        #ci_lower = sim_df['CANON_NONCANON_COSIM_CI'].apply(lambda x: x[0])
        #ci_upper = sim_df['CANON_NONCANON_COSIM_CI'].apply(lambda x: x[1])

        #ax.fill_between(sim_df['START_year'], ci_lower, ci_upper, alpha=0.2)
        label_fontsize = 16

    elif cosim_to_plot == 'TOTAL_COSIM_MEAN':
        ax.set_title(f'Total data ({col_or_grey}), $\\rho = {corr:.2f}$', fontsize = 14)

        # add error band
        CI = sim_df['TOTAL_COSIM_CI']
        ci_lower = CI.apply(lambda x: x[0])
        ci_upper = CI.apply(lambda x: x[1])
        ax.fill_between(sim_df['START_year'], ci_lower, ci_upper, alpha=0.2, color = color)

        # specify fontsize

        label_fontsize = 12

    elif cosim_to_plot == 'NONCANON_COSIM_MEAN':
        ax.set_title(f'Total non-canon ({col_or_grey}), $\\rho = {corr:.2f}$', fontsize = 12)
        
        # add error band
        CI = sim_df['NONCANON_COSIM_CI']
        ci_lower = CI.apply(lambda x: x[0])
        ci_upper = CI.apply(lambda x: x[1])
        ax.fill_between(sim_df['START_year'], ci_lower, ci_upper, alpha=0.2, color = color)

        # specify fontsize
        label_fontsize = 12

    else:
        ax.set_title(f'{title_mapping[canon_col]} ({col_or_grey}), $\\rho = {corr:.2f}$', fontsize = 17)

        # add error band
        CI = sim_df['CANON_COSIM_CI']
        ci_lower = CI.apply(lambda x: x[0])
        ci_upper = CI.apply(lambda x: x[1])
        ax.fill_between(sim_df['START_year'], ci_lower, ci_upper, alpha=0.2, color = color)

        label_fontsize = 16

    # create plot
    ax.set_xlabel('t', fontsize = label_fontsize)
    ax.set_ylabel(ylabel, fontsize = label_fontsize)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', which='major', length=4, width=1, labelsize=12)
    
def create_stacked_freqplot(df, ax, canon_col, w_size, year_col = 'start_year'):

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

    groupobject = df.groupby([year_col, canon_col]).size().unstack()
    groupobject = groupobject.rename(columns=column_mapping)
    
    groupobject.plot(kind='bar', stacked=True, ax=ax, edgecolor='none', width=0.95, color=[bar_colors.get(col, '#333333') for col in groupobject.columns])

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

    # Grid for clarity
    ax.grid(True, linestyle='--', alpha=0.5)

    # Final layout cleanup
    ax.margins(x=0)

def plot_grid(df, color_subset, canon_cols, w_size, cosim_to_plot, title, savefig, filename):

    fig, axs = plt.subplots(2, 3, figsize=(19, 11))

    for idx, col in enumerate(tqdm(canon_cols, desc=f"Plotting canon columns for {cosim_to_plot}")):

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

        if idx != 0:
            axs[0, idx].set_ylabel('')   # Remove Y label for all columns except for first
            axs[1, idx].set_ylabel('') 

    fig.tight_layout()

    if savefig:
            plt.savefig(filename, format='pdf', bbox_inches='tight')


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

def analysis_plots(df, color_subset, w_size, canon_cols):

    ################# plot pca_binary, greyscale and color embeddings #################
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
    
    plt.savefig(os.path.join('figs', f'PCA_exb_only.pdf'), format='pdf', dpi=300)

    ################# plot inter/intra group plots #################

        # canon
    plot_grid(df = df,
           color_subset=color_subset, 
            canon_cols = canon_cols,
            w_size= 30, 
            cosim_to_plot='CANON_COSIM_MEAN', 
            title='', 
            savefig=True,
            filename=os.path.join('figs', f'intra_canon_w30.pdf'))

        # intra, non-canon
    plot_grid(df = df, 
            color_subset = color_subset, 
            canon_cols = canon_cols,
            w_size= 30, 
            cosim_to_plot='NONCANON_COSIM_MEAN', 
            title='',
            savefig=True,
            filename=os.path.join('figs', f'intra_noncanon_w30.pdf'))
    
        # inter-group
    plot_grid(df = df, 
              color_subset = color_subset, 
              canon_cols = canon_cols,
              w_size= 30, 
              cosim_to_plot='CANON_NONCANON_COSIM', 
              title='',
              savefig=True,
              filename=os.path.join('figs', f'inter_w30.pdf'))
    
        # plot total non-canon data
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

    plt.savefig(os.path.join('figs', f'total_noncanon_w30.pdf'), format='pdf', dpi=300)

def pca_icons(ax, df, ds, embedding, image_col, out_folder, filename):
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

    if image_col == 'grey_image':
        def getImage(img):
            return OffsetImage(np.array(img), zoom=.02, cmap='gray') # need to change color map if plotting greyscale; matplotlib defaults 1-channel images to different cmap otherwise..
    
    else:
        def getImage(img):
            return OffsetImage(np.array(img), zoom=.02)

    for index, row in df_pca.iterrows():
        # add images to plot
        ab = AnnotationBbox(getImage(ds[index][image_col]), (row["PCA1"], row["PCA2"]), frameon=False)
        ax.add_artist(ab)

    plt.savefig(os.path.join(out_folder, filename), format='eps', dpi=1200)

    np.seterr(divide='ignore', invalid='ignore')

def umap_plot(ax, df, ds, embedding, filename, n_components=50):

    def getImage(img):
        return OffsetImage(np.array(img), zoom=.02)

    embeddings_array = np.array(df[embedding].to_list(), dtype=np.float32)

    # reduce dimensionality
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(embeddings_array)

    X = np.array(pca_results)
    umap_fitted = UMAP(n_components=2, random_state=42).fit_transform(X)
    df_umap = pd.DataFrame(umap_fitted, columns=["umap1", "umap2"])
    
    ax.scatter(df_umap['umap1'], df_umap['umap2'], color='white')

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.axis("equal")
    ax.set_axis_off()

    for index, row in df_umap.iterrows():
        # add images to plot
        ab = AnnotationBbox(getImage(ds[index]['image']), (row["umap1"], row["umap2"]), frameon=False)
        ax.add_artist(ab)

    plt.savefig(os.path.join('testy', filename), format='eps', dpi=1200)

    np.seterr(divide='ignore', invalid='ignore')

def plot_pca_comparison(df,ds,start_period=(1750, 1780),end_period=(1781, 1810), embedding_type='embedding', canon_filter='all', title=None, save_path=None):

    def filter_data(dataframe, year_range, canon):
        subset = dataframe.query(f'start_year >= {year_range[0]} and start_year <= {year_range[1]}')
        if canon != 'all':
            subset = subset.query(f'exb_canon == "{canon}"')
        idxs = subset.index.tolist()
        return subset, ds.select(idxs)

    df1, ds1 = filter_data(df, start_period, canon_filter)
    df2, ds2 = filter_data(df, end_period, canon_filter)

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    pca_icons(axs[0], df1, ds1, embedding_type, 'image', 'figs', 'trash1.pdf')
    pca_icons(axs[1], df2, ds2, embedding_type, 'image', 'figs', 'trash2.pdf')

    if title:
        fig.suptitle(title, fontsize=15)

    if save_path:
        plt.savefig(os.path.join('testy', save_path), bbox_inches='tight', format='pdf', dpi=1200)

def dataset_visualizations(canon_cols, df, ds, color_subset, ds_color):
    
    ################# plot canon/painting frequency #################
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    for idx, col in enumerate(canon_cols):
       create_stacked_freqplot(df = df, 
                               ax = axs[idx], 
                               canon_col = col, 
                               w_size = 30, 
                               year_col = 'start_year')
       if idx != 0:
           axs[idx].set_ylabel('')   # Remove Y label for all columns except for first
           axs[idx].legend().remove()
    
    plt.tight_layout()
    plt.savefig(os.path.join('testy', 'canon_frequency.pdf'), format='pdf', dpi=300)
    
    ################## plot PCA with painting icons #################

    fig, axs = plt.subplots(1, 1, figsize=(30, 20))

    print('Creating PCA plots....')
    pca_icons(axs, 
              color_subset, 
              ds_color, 
              'embedding', 
              'image', 
              'testy', 
              'pca_paitings_color.eps')

    # add column wiht greyscaled images
    grey_images = []

    feature = Image_ds(decode=False)

    for i in tqdm(range(len(ds))):
        image = ds[i]['image']
        image_grayscale = image.convert('L')
        image_encoded = feature.encode_example(image_grayscale) # in order to add an image to a HF dataset column, the image needs to be encoded properly
        grey_images.append(image_encoded)

    ds = ds.add_column('grey_image', grey_images)
    ds = ds.cast_column('grey_image', Image_ds(decode=True))

    pca_icons(axs, 
              df, 
              ds, 
              'grey_embedding', 
              'grey_image', 
              'testy', 
              'pca_paitings_grey.eps')

    plot_pca_comparison(color_subset,
                        ds_color, 
                        canon_filter='all', 
                        save_path='pca_innovation_total.pdf')

    plot_pca_comparison(color_subset,
                        ds_color, 
                        canon_filter='canon', 
                        save_path='pca_innovation_canon.pdf')
    
    plot_pca_comparison(color_subset,
                        ds_color, 
                        canon_filter='other', 
                        save_path='pca_innovation_non_canon.pdf')
    
    ################## plot and save color UMAP ##################
    fig, axs = plt.subplots(1, 1, figsize=(20, 15))
    print('Creating UMAP plot...')
    umap_plot(axs, color_subset, ds_color, 'embedding', 'umap_n50_color.eps')

def print_classification_results(canon_cols, models, sampling_methods, df, embedding_col, col_or_grey, out_folder):

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
    out_file = os.path.join(out_folder, f'{col_or_grey}_classification_report.txt')

    with open(out_file, 'w') as file:
                file.write(str(table))

