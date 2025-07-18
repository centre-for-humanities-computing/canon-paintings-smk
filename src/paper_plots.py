import argparse
import os
import matplotlib.pyplot as plt
#from analyses_utils import all_analyses_plots, print_classification_results
from .analyses_utils import pca_binary, plot_grid, pca_icons, create_stacked_freqplot, umap_plot, plot_exb_venues, plot_diachronic_change, plot_pca_comparison
import datasets

import pandas as pd

def argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--df_name', type=str, help='name of pandas df in /data folder to use')
    parser.add_argument('--ds_name', type=str, help='name of huggingface ds in /data folder to use')

    args = vars(parser.parse_args())
    
    return args

def main():
    
    args = argument_parser()

    # read df from pickle 
    df = pd.read_pickle(os.path.join('data', args['df_name']))
    ds = datasets.load_from_disk(os.path.join('data', args['ds_name']))

    # create total canon/non-canon variable

    # plot intra-group for all non-canon data
    total_canon = []
    for idx, row in df.iterrows():
        if (
            row['exb_canon'] == 'canon' or
            row['smk_exhibitions'] == 'canon' or
            row['on_display'] == 'canon'
        ):
            total_canon.append('canon')
        else:
            total_canon.append('other')

    df['total_canons'] = total_canon 

    # make subset of colored images only 
    color_subset = df.query('rgb == "color"')
    color_idx = color_subset.index.tolist()
    ds_color = ds.select(color_idx)
    color_subset.reset_index(drop=True, inplace=True)

    # plot pca_binary, greyscale and color embeddings
    #fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    #pca_binary(ax = axs[0], 
                #df = color_subset, 
                #embedding = 'embedding', 
                #canon_category = 'exb_canon', 
                #title = "Exhibition canon (color)")
    
    #pca_binary(ax = axs[1], 
        #df = df, 
        #embedding = 'grey_embedding', 
        #canon_category = 'exb_canon', 
        #title = "Exhibition canon (greyscale)")
    
    #plt.savefig(os.path.join('figs', f'PCA_exb_only.pdf'), format='pdf', dpi=300)

    # plot inter/intra group without frequency plots

    # define canon columns
    canon_cols = ['exb_canon', 'smk_exhibitions', 'on_display']

    # canon
    #plot_grid(df = df,
           #color_subset=color_subset, 
            #canon_cols = canon_cols,
            #w_size= 30, 
            #cosim_to_plot='CANON_COSIM_MEAN', 
            #title='', 
            #savefig=True,
            #filename=os.path.join('figs', f'intra_canon_w30.pdf'))

        # intra, non-canon
    #plot_grid(df = df, 
            #color_subset = color_subset, 
            #canon_cols = canon_cols,
            #w_size= 30, 
            #cosim_to_plot='NONCANON_COSIM_MEAN', 
            #title='',
            #savefig=True,
            #filename=os.path.join('figs', f'intra_noncanon_w30.pdf'))
    
        # inter-group
    #plot_grid(df = df, 
           #   color_subset = color_subset, 
          #    canon_cols = canon_cols,
         #   w_size= 30, 
        #        cosim_to_plot='CANON_NONCANON_COSIM', 
       #         title='',
      #          savefig=True,
     #           filename=os.path.join('figs', f'inter_w30.pdf'))
    
        # plot total non-canon data
    #fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    #plot_diachronic_change(w_size = 30, 
    #                df = color_subset, 
    #                canon_col = 'total_canons', 
    #                embedding_col = 'embedding', 
    #                cosim_to_plot = 'NONCANON_COSIM_MEAN', 
    #                ax = axs[0])
    
    #plot_diachronic_change(w_size = 30, 
         #       df = df, 
        #        canon_col = 'total_canons', 
       #         embedding_col = 'grey_embedding', 
      #          cosim_to_plot = 'NONCANON_COSIM_MEAN', 
     #           ax = axs[1])
    
    # remove y label from second plot
    #axs[1].set_ylabel("")

    #plt.savefig(os.path.join('figs', f'total_noncanon_w30.pdf'), format='pdf', dpi=300)

    # plot PCA with painting icons

    fig, axs = plt.subplots(1, 1, figsize=(30, 20))

    #print('Creating PCA plot....')
    #pca_icons(axs, color_subset, ds_color, 'embedding', 'image', 'figs', 'pca_paitings_color.eps')

    #print('Creating PCA plot....')
    pca_icons(axs, df, ds, 'grey_embedding', 'grey_image', 'figs', 'pca_paitings_grey.eps')

    # plot canon/painting frequency 
    #fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    #for idx, col in enumerate(canon_cols):
     #   create_stacked_freqplot(df = df, 
      #                          ax = axs[idx], 
       #                         canon_col = col, 
        #                        w_size = 30, 
         #                       year_col = 'start_year')
        #if idx != 0:
         #   axs[idx].set_ylabel('')   # Remove Y label for all columns except for first
          #  axs[idx].legend().remove()

    #plt.tight_layout()
    #plt.savefig(os.path.join('figs', 'canon_frequency.pdf'), format='pdf', dpi=300)

    # plot and save color UMAP
    #fig, axs = plt.subplots(1, 1, figsize=(20, 15))
    #print('Creating UMAP plot...')
    #umap_plot(axs, color_subset, ds_color, 'embedding', 'umap_n50_color.eps')

    #plot_exb_venues(df, 'exb_venues.pdf')

    # total data
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
    
if __name__ == '__main__':
   main()