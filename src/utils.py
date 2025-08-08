from PIL import Image
import os
import pandas as pd
import numpy as np
import datasets
from tqdm import tqdm
from datasets import Dataset
from datasets import load_dataset
import matplotlib.pyplot as plt 
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib.patches import Patch
from sklearn.metrics.pairwise import cosine_distances
import seaborn as sns

def find_neighbors(feature_list, target_image):
    # initialize K-nearest neighbors algorithm
    neighbors = NearestNeighbors(n_neighbors=10, 
                            algorithm='brute',
                            metric='cosine').fit(feature_list)
    
    # save the indices and distances of the neighbors to the target image
    distances, indices = neighbors.kneighbors([feature_list[target_image]])

    # initialize empty lists
    idxs = []
    dist = []
    
        # save the 5 closest images' indices and distances
    for i in range(1,10):
        idxs.append(indices[0][i])
        dist.append(distances[0][i])

    # create dataframe
    data = pd.DataFrame({
                        "distance_score" : pd.Series(dist),
                        'index': pd.Series(idxs)})
    
    print(data['index'])
    
    # return filenames as a pandas series to be used in the plotting function
    return data
def show_plot(names, target_image, dataset):
    
    # arrange plots
    f, axarr = plt.subplots(3, 3,figsize=(8, 6))
    
    
    # print target image
    axarr[0,0].imshow(dataset[target_image]['image'])
    axarr[0, 0].set_title('Target Image')

    # plot 5 most similar next to it
    axarr[0,1].imshow(dataset[names[0]]['image'])
    axarr[0,2].imshow(dataset[names[1]]['image'])
    axarr[1,0].imshow(dataset[names[2]]['image'])
    axarr[1,1].imshow(dataset[names[3]]['image'])
    axarr[1,2].imshow(dataset[names[4]]['image'])
    axarr[2,0].imshow(dataset[names[5]]['image'])
    axarr[2,1].imshow(dataset[names[6]]['image'])
    axarr[2,2].imshow(dataset[names[7]]['image'])
    
    # remove axes from plot
    #for ax in f.axes:
      #  ax.axison = False

    for ax in f.axes:
        ax.axis('off')
        ax.set_aspect('equal')

    f.subplots_adjust(wspace=0.01, hspace=0.01)
        
    #plt.show()
    
    #plt.subplots_adjust(wspace=0, hspace=0)
    #plt.subplots_adjust(left=0, bottom=0, right=1, top=0, wspace=0, hspace=0)
    #plt.savefig(os.path.join('out', 'plots', plot_name))
    plt.show()
  
def plot_neighbors(feature_list, target_image, dataset):

    # find closest images and save in a df
    data = find_neighbors(feature_list, target_image)

    # save the indices of the closest images
    indices = data['index'].tolist()

    # plot them
    show_plot(indices, target_image, dataset)

def plot_pca(ax, data, title, colormapping):
    # Handle embeddings
    embeddings_array = np.array(data["embedding"].to_list(), dtype=np.float32)

    # Make labels titlecase
    colormapping = {k.replace('_', ' ').title(): v for k, v in colormapping.items()}
    # Replace 'o' with 'Other'
    
    # to 2 dimensions
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(embeddings_array)
    
    df_pca = pd.DataFrame(pca_results, columns=["PCA1", "PCA2"])
    
    # Add metadata
    df_pca["canon"] = data["canon"].values
    df_pca["canon"] = df_pca["canon"].apply(lambda x: x.replace('_', ' ').title())
    # replace 'O' with 'Other'
    #df_pca["category"] = df_pca["category"].apply(lambda x: 'Other' if x == 'O' else x)


    # We're gonna set a different alpha for the 'O' category
    alpha_dict = dict(zip(colormapping.keys(), [0.65 if x != 'other' else 0.2 for x in colormapping.keys()]))
    # Update color dict to have titlecase

    # Plot each category
    for category in df_pca["canon"].unique():
        subset = df_pca[df_pca["canon"] == category]

        #marker = markers_dict.get(category) 
        alpha = alpha_dict.get(category)
        
        ax.scatter(
            subset["PCA1"],
            subset["PCA2"],
            color=colormapping.get(category),
            label=category,
            alpha=alpha,
            edgecolor='black',
            s=110,
            marker='o' #marker
        )

    ax.set_title(title)
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")

    legend_handles = [Patch(facecolor=color, label=label) for label, color in colormapping.items()]
    ax.legend(handles=legend_handles, loc='upper right')

    ax.axis("equal")

import numpy as np
from numpy.typing import ArrayLike

def calc_vector_histogram(x: ArrayLike, bins: int = 256) -> np.ndarray:
    # calculate a histogram
    hist, bin_edges = np.histogram(x, bins=bins)
    # normalize histogram to sum to one
    hist_norm = hist / hist.sum()

    return hist_norm

"""
"""
from typing import Union
import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics.pairwise import cosine_similarity


def _rel_entr(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Vectorized, element-wise relative entropy.
    """
    # add smalles possible float to avoid RunTime warnings
    # about zero division
    p = np.asarray(p) + np.finfo(float).eps
    q = np.asarray(q) + np.finfo(float).eps

    mask = (p > 0) & (q > 0)
    result = np.where(mask, p * np.log2(p / q), 0)
    return result


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    """
    Kullback-Leibler Divergence between two discrete distributions.
    """
    return _rel_entr(p, q).sum(axis=-1)


def js_div(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon divergence between two discrete distributions.
    """
    p = p / np.sum(p, axis=-1, keepdims=True)
    q = q / np.sum(q, axis=-1, keepdims=True)
    m = (p + q) / 2.0
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def js_dist(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jesen-Shannon *distance* between two discrete distributions.
    """
    return np.sqrt(js_div(p, q))


def cos_sim(x: np.ndarray, y: np.ndarray) -> Union[float, np.ndarray]:
    """Wrapper of sklearn's cosine_similarity that handles 1d arrays."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    return cosine_similarity(x, y)


DISTANCES = {
    "kld": kl_div,
    "jsd": js_div,
    "jensenshannon": js_dist,
    "cosine": cos_sim,
}

"""
Class for estimation of information dynamics of time-dependent probabilistic document representations.

TODO
- window_size=1 doesn't work
"""
import warnings
from typing import Optional, Union, Callable, Tuple, Literal
from typing_extensions import Self

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin


class WindowedRollingDistance(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        measure: Union[str, Callable],
        window_size: int,
        jump_size: int = 0,
        inplace_constant: Union[int, float] = 0,
        estimate_error: bool = False,
        averaging_weigts: Optional[ArrayLike] = None,
    ):
        """

        Parameters
        -----------
        measure : str or Callable
            distance / similarity metric to use for comparison of observations.
            Either a function, or a string 

        window_size : int
            context size (n observations) for calculating novelty.

        jump_size : int, default=0
            number of observations to ignore before window kicks in.

        inplace_constant : int or float, default=0
            constant number that gets added to the signal at the beginning and end of the time series.

        estimate_error: bool, default=False
            Return the standard deviation of the novelty, transience and resonance signals?

        averaging_weigts : np.ndarray or None, default=None
            Weights for averaging novelty and transience.
            Can be used to give more weight to more recent observations.
            Must be of length window - 1.
        """
        self.window_size = window_size
        self.jump_size = jump_size
        self.inplace_constant = inplace_constant
        self.estimate_error = estimate_error
        self.averaging_weigts = averaging_weigts

        if measure in DISTANCES:
            self.measure = DISTANCES[measure]
        elif callable(measure):
            self.measure = measure
        else:
            raise AttributeError(f"Invalid value for parameter 'measure': {measure}")

    def _summarize_tmp(self, tmp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ """
        if self.averaging_weigts is not None:
            return np.average(tmp, weights=self.averaging_weigts, axis=0), np.std(tmp)
        else:
            return np.mean(tmp), np.std(tmp)

    def _validate_data(self, X: ArrayLike) -> np.ndarray:
        """
        UNFINISHED
        Super basic data validation, converts lists of lists to np.ndarray.
        """
        if isinstance(X, list):
            if isinstance(X[0], list):
                X = np.array(X)
        elif isinstance(X, np.ndarray):
            pass
        else:
            warnings.warn(
                f"Input data type ({type(X)}) not recognized. Returning input data as is."
            )

        return X

    def calc_signal(
        self, X: np.ndarray, relative_to: Literal['past', 'future']
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Iterates over points in X and calculates the distance between each point 
        and it's context (past or future points).
        
        Parameters
        ----------
        X : {array-like, pd.DataFrame, List[List[float]]}
            of shape (n_samples, n_features)

        relative_to : Literal['past', 'future']
            calculate distance between i and
            past: previous elements in X
            future: following elements in X
        """

        ts_len = X.shape[0]

        H_hat = np.empty(ts_len)
        H_sd = np.empty(ts_len)

        for i, x in enumerate(X):
            # select {window} elements after i, or empty array
            if relative_to == "past":
                jump_index = i - self.jump_size
                # compare x with elements before it
                i_lower = max(0, jump_index - self.window_size)
                i_upper = max(0, jump_index)
                submat = X[i_lower:i_upper,]
            elif relative_to == "future":
                jump_index = i + self.jump_size
                # compare x with elements after it
                i_lower = jump_index
                i_upper = min(ts_len - 1, jump_index + self.window_size)
                submat = X[i_lower:i_upper,]
            else:
                raise ValueError(
                    f"Invalid value for parameter 'relative_to': {relative_to}. Must be 'past' or 'future'."
                )

            if submat.shape[0] == self.window_size:
                # use broadcasting to calculate js_dist for all elements in submat
                tmp = self.measure(x, submat)
                # skip the first comparison (x with itself)
                tmp = tmp[1:]
            else:
                tmp = np.full([self.window_size - 1], self.inplace_constant)

            H_hat[i], H_sd[i] = self._summarize_tmp(tmp)

        return H_hat, H_sd

    def _calc_resonance(
        self, N_hat: np.ndarray, T_hat: np.ndarray, N_sd: np.ndarray, T_sd: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ """
        R_hat = N_hat - T_hat
        R_sd = (N_sd + T_sd) / 2

        # invalidate the signal outside of window bounds
        # R_hat[: self.window_size] = np.zeros([self.window_size], dtype=float) + self.inplace_constant
        # R_hat[-self.window_size :] = np.zeros([self.window_size], dtype=float) + self.inplace_constant
        # R_sd[: self.window_size] = np.zeros([self.window_size], dtype=float) + self.inplace_constant
        # R_sd[-self.window_size :] = np.zeros([self.window_size], dtype=float) + self.inplace_constant

        return R_hat, R_sd

    def fit(self, X: ArrayLike, y=None) -> Self:
        """Fit signal to data: distance from past, distance from future, difference between distances.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to estimate fit the signal on.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Instance with estimated signal.
        """
        X = self._validate_data(X)

        # novelty & transience
        N_hat, N_sd = self.calc_signal(X, relative_to="past")
        T_hat, T_sd = self.calc_signal(X, relative_to="future")
        # resonance
        # TODO consider turning this off for jump calculations
        R_hat, R_sd = self._calc_resonance(N_hat, T_hat, N_sd, T_sd)

        # base output
        signal = {}

        if self.estimate_error:
            signal.update(
                {
                    "N_hat": N_hat,
                    "N_sd": N_sd,
                    "T_hat": T_hat,
                    "T_sd": T_sd,
                    "R_hat": R_hat,
                    "R_sd": R_sd,
                }
            )
        else:
            signal.update(
                {
                    "N_hat": N_hat,
                    "T_hat": T_hat,
                    "R_hat": R_hat,
                }
            )

        self.signal = signal
        return self

    def fit_transform(self, X: ArrayLike, y=None) -> dict[str, np.ndarray]:
        """
        Fit to data, return dict with signal.

        Parameters
        ----------
        X : {array-like, pd.DataFrame, List[List[float]]}
            of shape (n_samples, n_features)
            Training set.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        self.fit(X)
        return self.signal

def generate_xticks(df):
    # flag indices, where the next element is different
    yr_switches = np.diff(df["year"])
    # add last element
    yr_switches = np.append(0, yr_switches)
    # convert to bool
    yr_switches = np.array(yr_switches, dtype=bool)

    # get index (x-axis) and year variables (x-axis label)
    sig_index = np.array(df.index)
    sig_year = np.array(df["year"])

    # mark changes in year
    xticks_idx = sig_index[yr_switches]
    xticks_label = sig_year[yr_switches]

    return xticks_idx, xticks_label

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

# also create plot to handle canon variables as continuous scales (i.e., they are colored differently)
def plot_pca_scale(ax, df, embeddings_column, color_by, title, cmap):
    
    # Handle embeddings
    embeddings_array = np.array(df[embeddings_column].to_list(), dtype=np.float32)
    
    # to 2 dimensions
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(embeddings_array)
    
    df_pca = pd.DataFrame(pca_results, columns=["PCA1", "PCA2"])
    
    # Add metadata
    df_pca["canon"] = df[color_by].values

    # Plot each category
    scatterplot = ax.scatter(df_pca["PCA1"], df_pca["PCA2"], c=df_pca["canon"], marker='o', cmap=cmap, alpha = 0.4)
    ax.set_title(title)
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.axis("equal")

    return scatterplot

def plot_dendrogram(df, col_to_color, col_to_label, embedding_col, l, h, palette='Set2'):
    
    import seaborn as sns

    df[col_to_color] = df[col_to_color].replace({0: 'other', 1: 'canon'})

    unique_categories = df[col_to_color].unique()

    # colors
    cat_map = dict(zip(df[col_to_label],df[col_to_color]))
    color_dict = {'other': '#129525', 'canon': '#356177'}#'#FCCA46'}

    # prepare data for plotting
    embeddings_matrix = np.stack(df[embedding_col].values)
    cosine_dist_matrix = cosine_distances(embeddings_matrix)
    
    if cosine_dist_matrix.shape[0] != cosine_dist_matrix.shape[1]:
        raise ValueError("Distance matrix is not square.")

    Z = linkage(cosine_dist_matrix, method='ward')

    # dendrogram plot
    sns.set_style('whitegrid')
    plt.figure(figsize=(l, h))
    dend = dendrogram(Z, labels=df[col_to_label].values, orientation='top', leaf_font_size=5, color_threshold=0, above_threshold_color='black')

    # Labels
    # get x-tick labels
    ax = plt.gca()
    xticklabels = ax.get_xticklabels()

    # apply colors labels
    used_colors = {}
    for tick in xticklabels:
        label = tick.get_text()
        # just to make sure we have no other labels in there
        if label in cat_map:
            value = cat_map[label]
            color = color_dict[value]
            tick.set_color(color)
            used_colors[value] = color
        else:
            tick.set_color('black')
    
    # update labels in used_colors, make titlecase
    used_colors = {k.replace('_', ' ').title(): v for k, v in used_colors.items()}
    # make "other" if O in used_colors
    if 'O' in used_colors:
        used_colors['Other'] = used_colors.pop('O')
    
    # layout
    plt.xlabel("Cosine Distance")

    legend_handles = [Patch(facecolor=color, label=label) for label, color in used_colors.items()]
    ax.legend(handles=legend_handles, loc='upper right', size = 10)

    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()

def plot_bar_dendrogram(df, col_to_color, embedding_col, l, h, title):
    
    # define colors for 'other' and 'canon' category
    color_dict = {'other': '#129525', 'canon': '#356177'}

    # map colors to canon variable
    col_colors = df[col_to_color].map(color_dict).to_numpy()

    # Prepare embeddings and distances
    embeddings_matrix = np.stack(df[embedding_col].values)
    cosine_dist_matrix = cosine_distances(embeddings_matrix)

    if cosine_dist_matrix.shape[0] != cosine_dist_matrix.shape[1]:
        raise ValueError("Distance matrix is not square.")

    # calculate linkage matrix for hiearchical clustering of painting embeddings
    Z = linkage(cosine_dist_matrix, method='ward')

    # create dendrogram
    # we are using seaborn to plot the colors as a bar
    # seaborn does however plot their dendrograms vertically, so we need to transpose the matrix and color by columns to plot horizontally 
    sns.set_theme(color_codes=True)

    g = sns.clustermap(
        embeddings_matrix.T, # transpose matrix so it's now (dimensions, paintings)
        col_colors=col_colors, # color by columns (i.e., canon category for painting)
        row_cluster=False,               
        col_linkage=Z, # use custom linkage matrix
        dendrogram_ratio=(0.01, 0.9), 
        colors_ratio=0.04,
        figsize=(l, h)
    )

    # remove heatmap (it's just a binary variable so not much information here)
    g.ax_heatmap.remove()
    g.cax.remove()

    # add legend
    handles = [Patch(color=color_dict[key], label=key) for key in color_dict]
    g.ax_col_dendrogram.legend(
        handles=handles,
        loc = 'upper right',
        fontsize = '10',
        facecolor = 'none', # remove grey background from legend box
        title= 'Canon label'
    )

    # adjust layot of plot
    plt.subplots_adjust(top=0.98, bottom=0.15, left=0.05, right=0.95, hspace=0.05)

    # axes are a bit messed up so I'm just adding a y axis manually using a text box
    g.fig.text(0.02, 0.6, "Cosine Distance", va='center', rotation='vertical', fontsize=12)
    
    # add title
    g.fig.suptitle(title, y=1, fontsize=16)

    plt.show()


