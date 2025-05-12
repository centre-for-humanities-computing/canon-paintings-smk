source ./env/bin/activate

# extract embeddings for paintings as they are
python3 src/smk_embeddings.py --dataset louisebrix/smk_all_paintings --model eva02_large_patch14_clip_336.merged2b --npy_file eva02_clip_all_paintings.npy

# convert all paintings to greyscale and extract embeddings
python3 src/greyscale_embeddings.py --dataset louisebrix/smk_all_paintings --model eva02_large_patch14_clip_336.merged2b --npy_file GREY_eva02_clip_all_paintings.npy

deactivate