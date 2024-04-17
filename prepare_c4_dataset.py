import os
import datasets

# Check the paths used in this script, and adjust them if necessary.
dataset_dir = '/data/shared_data/c4'
temp_dir    = '/data/shared_data/c4_tmp'

os.system('mkdir -p ' + temp_dir)
os.system('mkdir -p ' + dataset_dir)

print('==> Downloading...')
data = datasets.load_dataset(
    'allenai/c4'            ,
    'en'                    ,
    cache_dir   = temp_dir  ,
    streaming   = False     ,
    num_proc    = 32
)
print(data)

print('\n==> Saving...')
data.save_to_disk(dataset_dir, num_proc=32)

print('\n==> Checking...')
loaded_data = datasets.load_from_disk(dataset_dir)
print(loaded_data)

os.system('rm -rf ' + temp_dir)
