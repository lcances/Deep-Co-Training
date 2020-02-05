from datasetManager import DatasetManager
from generators import Generator

import matplotlib.pyplot as plt
import numpy as np
import time
import tqdm
import librosa.display

def timit(func):
    def decorator(*args, **kwargs):
        start_time = time.time()
        value = func(*args, **kwargs)
        end_time = time.time() - start_time
        
        print("run in %.4fs" % end_time)
        return value
    return decorator

audio_root = "../dataset/audio"
metadata_root = "../dataset/metadata"
dataset = DatasetManager(metadata_root, audio_root, train_fold=[1], val_fold=[2])

# test generator train / val
train_loader = Generator(dataset, train=True, val=False, sampling=1.0, augments=[])
val_loader = Generator(dataset, train=False, val=True, sampling=1.0, augments=[])

# test really different
print(np.array(train_loader.y.index.values) == np.array(val_loader.y.index.values))

# Test function of the cache
@timit
def extract_all_files(loader):
    for i in tqdm.tqdm(range(len(loader))):
        x = loader[i]
        
extract_all_files(train_loader)
extract_all_files(train_loader)
extract_all_files(val_loader)
extract_all_files(val_loader)


