import argparse
import os
import datasets
from datasets import load_dataset
import matplotlib.pyplot as plt
from src.analyses_utils import analysis_plots, dataset_visualizations, save_classification_results, add_image_col
import pandas as pd

def argument_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', type=str, help='name of dataset from HuggingFace hub')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, help='whether to plot results or not', default=False)
    parser.add_argument('--classification', action=argparse.BooleanOptionalAction, help='whether to run classification or not', default=False)

    args = vars(parser.parse_args())
    
    return args

def main():

    args = argument_parser()

    # load data
    ds = load_dataset(args['ds_name'], split="train")

    # download the images from SMK using the links from the 'image_thumbnail' column
    ds = add_image_col(ds, 'image_thumbnail')
    
    # convert to pd dataframe
    df = ds.to_pandas()

    # make subset of colored images only 
    color_subset = df.query('rgb == "color"')
    color_idx = color_subset.index.tolist()
    ds_color = ds.select(color_idx)
    color_subset.reset_index(drop=True, inplace=True)

    # define canon columns
    canon_cols = ['exb_canon', 'smk_exhibitions', 'on_display']

    if args['plot']:

        print('Creating plots....')

        analysis_plots(df=df, 
                       color_subset = color_subset, 
                       w_size = 30, 
                       canon_cols= canon_cols)
        
        dataset_visualizations(canon_cols = canon_cols, 
                               df = df, 
                               ds = ds, 
                               color_subset = color_subset,
                               ds_color = ds_color)

    if args['classification']:

        print('Running classifiers....')

        models = ['logistic', 'mlp']
        sampling_methods = [False, True]

        # run for both greyscale and color data
        save_classification_results(canon_cols=canon_cols, 
                                     models=models, 
                                     sampling_methods=sampling_methods, 
                                     df=df, 
                                     embedding_col='grey_embedding', 
                                     col_or_grey='greyscale',
                                     save_report=True)

        save_classification_results(canon_cols=canon_cols, 
                                     models=models, 
                                     sampling_methods=sampling_methods, 
                                     df=color_subset, 
                                     embedding_col='embedding', 
                                     col_or_grey='color',
                                     save_report=True)

if __name__ == '__main__':
   main()

    