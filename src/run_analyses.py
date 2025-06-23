import argparse
import os
import matplotlib.pyplot as plt
from analyses_utils import plot_diachronic_change, plot_grid, plot_canon_novelty

def argument_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='name of HuggingFace Hub dataset containing images')
    parser.add_argument('--model', type=str, help='name of pretrained model from timm library to use')
    parser.add_argument('--npy_file', type=str, help ='what to call saved npy files with embeddings')
    args = vars(parser.parse_args())
    
    return args

def all_analyses_plots(df, plot_suffix, plot_folder, inter_intra_w, novelty_w):

    # make subset of colored images only 
    color_subset = df.query('rgb == "color"')
    #color_idx = color_subset.index.tolist()
    #ds_color = ds.select(color_idx)
    color_subset.reset_index(drop=True, inplace=True)

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
    plt.savefig(os.path.join('..', plot_folder, f'intra_total_w{inter_intra_w}_{plot_suffix}.png'), format='png')

    # intra, canon
    canon_cols = ['exb_canon', 'smk_exhibitions', 'on_display']

    plot_grid(canon_cols = canon_cols,
                w_size= inter_intra_w, 
                cosim_to_plot='CANON_COSIM_MEAN', 
                title='Intra-group analysis for all canon variables', 
                savefig=True,
                filename=os.path.join('..', plot_folder, f'intra_canon_w{inter_intra_w}_{plot_suffix}.png'))
    
    # intra, non-canon
    plot_grid(canon_cols = canon_cols,
        w_size= inter_intra_w, 
        cosim_to_plot='NONCANON_COSIM_MEAN', 
        title='Intra-group analysis for all non-canon',
        savefig=True,
        filename=os.path.join('..', plot_folder, f'intra_noncanon_w{inter_intra_w}_{plot_suffix}.png'))

    # inter-group
    plot_grid(canon_cols = canon_cols,
        w_size= inter_intra_w, 
        cosim_to_plot='CANON_NONCANON_COSIM', 
        title='Inter-group analysis for all canon variables',
        savefig=True,
        filename=os.path.join('..', plot_folder, f'inter_w{inter_intra_w}_{plot_suffix}.png'))

    # NOVELTY // ENTROPY PLOTS

    #mean_embedding_df_greyscale = mean_embedding_per_year(df, 'grey_embedding', novelty_w)
    #mean_embedding_df_color = mean_embedding_per_year(color_subset, 'embedding', novelty_w)

    # total data
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    plot_canon_novelty(df, 'total', 'none', 'embedding', novelty_w, 'start_year', 'greyscale', axs[0])
    plot_canon_novelty(color_subset, 'total', 'none', 'embedding', novelty_w, 'start_year', 'colored', axs[1])
    fig.suptitle('Novelty signal for total data with mean embedding per year (all data)', size = 13, y=0.97)
    plt.savefig(os.path.join('..', plot_folder, f'novelty_w=mean_embed_year_w{novelty_w}_{plot_suffix}.png'), format='png')

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
    plt.savefig(os.path.join('..', plot_folder, f'novelty_w=mean_embed_year_canon_w{novelty_w}_{plot_suffix}.png'), format='png')
    
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
    plt.savefig(os.path.join('..', plot_folder, f'novelty_w=mean_embed_year_noncanon_w{novelty_w}_{plot_suffix}.png'), format='png')

def run_supervised_classifications():

    