'''
Convert all images to greyscale and extract embeddings from greyscaled images
'''

import os 
import timm
import torch
import datasets
from datasets import load_dataset
from datasets import Dataset
from tqdm import tqdm
import cv2 
import numpy as np
from PIL import Image
import argparse
from typing import Any

def argument_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='name of HuggingFace Hub dataset containing images')
    parser.add_argument('--model', type=str, help='name of pretrained model from timm library to use')
    parser.add_argument('--npy_file', type=str, help ='what to call saved npy files with embeddings')
    args = vars(parser.parse_args())
    
    return args

def transform_and_extract(img:Image.Image, model:Any):

    '''
    Preprocess image and extract feature

    Parameters:
        - img: PIL image
        - model: Pretrained timm model
    
    Returns: 
        - feature_list_unnest: Image Embedding

    '''

    # extract information about transformations
    data_config = timm.data.resolve_model_data_config(model)

    # use this information to transform the data
    transforms = timm.data.create_transform(**data_config, is_training=False)
    
    #convert image to greyscale using standard transform (see PIL docs for specifications: https://pillow.readthedocs.io/en/stable/reference/Image.html)
    image_grayscale = img.convert('L')

    # convert back to 'color'. Image will still look black/white but will have three channels (the pixel's greyscale value is just multiplied for each of the color channels, i.e., it will have the same R, G and B value)
    # this is necessary as timm expects tensors with 3 channels
    image_rgb = image_grayscale.convert('RGB')

    # apply transformations, convert to tensor and extract features
    features = model(transforms(image_rgb).unsqueeze(0)) # unsqueeze adds a dim so the shape is now (1, 3, img_size, img_size)

    # convert from tensor to list
    feature_list = features.tolist()

    # un-nest list
    feature_list_unnest = feature_list[0]

    return feature_list_unnest

def features_from_dataset(dataset:Dataset, model:Any):
    '''
    Get image embeddings of all images in a dataset

    Parameters:
        - dataset: HuggingFace dataset with 'image' column
        - model: Pretrained timm model

    Returns:
        - embeddings: list of embeddings for all images
    '''

    # initialize empty list
    embeddings = []

    # loop over each image in the dataset and extract feature embeddings
    for i in tqdm(range(len(dataset)), desc="Extracting features from images"):
        try:
            image = dataset[i]['image']
            feature = transform_and_extract(image, model)
            embeddings.append(feature)
        except Exception as e:
            print(f"Error processing image {i}: {e}")

    return embeddings

def main():
    # parse arguments
    args = argument_parser()

    ds = load_dataset(args['dataset'], split="train")

    # initialize model
    model = timm.create_model(args['model'], pretrained=True, num_classes=0)
    model.eval() # turn on evaluation mode

    embeddings = features_from_dataset(ds, model)
    embeddings = np.array(embeddings)

    np.save(os.path.join('data', args['npy_file']), embeddings, allow_pickle=True)

if __name__ == '__main__':
   main()