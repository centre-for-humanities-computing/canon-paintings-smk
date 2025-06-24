import argparse
import os
import matplotlib.pyplot as plt
from analyses_utils import all_analyses_plots, print_classification_results
import pandas as pd

def argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--df_name', type=str, help='name of pandas df in /data folder to use')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, help='whether to plot results or not', default=False)
    parser.add_argument('--classification', action=argparse.BooleanOptionalAction, help='whether to run classification or not', default=False)
    parser.add_argument('--results_suffix', type=str, help='what suffix to add to plots')

    args = vars(parser.parse_args())
    
    return args

def main():

    args = argument_parser()

    # read df from pickle 
    df = pd.read_pickle(os.path.join('data', args['df_name']))

    # make subset of colored images only 
    color_subset = df.query('rgb == "color"')
    color_subset.reset_index(drop=True, inplace=True)

    # define canon columns
    canon_cols = ['exb_canon', 'smk_exhibitions', 'on_display']

    if args['plot']:

        all_analyses_plots(df=df, 
                           color_subset=color_subset, 
                           canon_cols=canon_cols, 
                           plot_suffix=args['results_suffix'], 
                           plot_folder='figs', 
                           inter_intra_w=30, # window size for inter/intra plots
                           novelty_w=5) # window size for novelty plots
    
    if args['classification']:

        models = ['logistic', 'mlp']
        sampling_methods = [False, True]

        # run for both greyscale and color data
        print_classification_results(canon_cols=canon_cols, 
                                     models=models, 
                                     sampling_methods=sampling_methods, 
                                     df=df, 
                                     embedding_col='grey_embedding', 
                                     col_or_grey='greyscale', 
                                     report_suffix=args['results_suffix'], 
                                     out_folder='figs')

        print_classification_results(canon_cols=canon_cols, 
                                     models=models, 
                                     sampling_methods=sampling_methods, 
                                     df=df, 
                                     embedding_col='embedding', 
                                     col_or_grey='color', 
                                     report_suffix=args['results_suffix'], 
                                     out_folder='figs')

if __name__ == '__main__':
   main()

    