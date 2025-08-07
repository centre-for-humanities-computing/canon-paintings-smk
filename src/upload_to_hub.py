import huggingface_hub
import datasets
import argparse
import os

def argument_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='name of huggingface dataset to upload')
    parser.add_argument('--hub_name', type=str, help='what to call the dataset uploaded to hub')
    args = vars(parser.parse_args())
    
    return args


def main():
    # parse argument
    args = argument_parser() 

    # load from disk
    ds = datasets.load_from_disk(os.path.join('data', args['dataset']))

    # push to CHC HuggingFace Hub
    ds.push_to_hub(f"chcaa/{args['hub_name']}")

if __name__ == '__main__':
   main()

