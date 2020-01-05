# -*- coding: utf-8 -*-
import os
#%%
# change this to your location
BOXCARS_DATASET_ROOT = "D:\\Master TAID\\Anul2\\MLAV\\Car-Detection-Mask-R-CNN\\DataSet\\BoxCars116k" 

#%%
BOXCARS_IMAGES_ROOT = os.path.join(BOXCARS_DATASET_ROOT, "images")
BOXCARS_DATASET = os.path.join(BOXCARS_DATASET_ROOT, "dataset.pkl")
BOXCARS_ATLAS = os.path.join(BOXCARS_DATASET_ROOT, "atlas.pkl")
BOXCARS_CLASSIFICATION_SPLITS = os.path.join(BOXCARS_DATASET_ROOT, "classification_splits.pkl")

