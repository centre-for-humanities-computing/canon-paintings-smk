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
from sklearn.decomposition import PCA
from matplotlib.patches import Patch

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
