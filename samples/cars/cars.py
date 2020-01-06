# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils


from utils import load_cache
import cv2
import os
#%%
# change this to your location
BOXCARS_DATASET_ROOT = "C:\\Users\\Nitroi\\Desktop\\Car-Detection-Mask-R-CNN\\Datasets\\BoxCars116k"

#%%
BOXCARS_IMAGES_ROOT = os.path.join(BOXCARS_DATASET_ROOT, "images")
BOXCARS_DATASET = os.path.join(BOXCARS_DATASET_ROOT, "dataset.pkl")
BOXCARS_ATLAS = os.path.join(BOXCARS_DATASET_ROOT, "atlas.pkl")
BOXCARS_CLASSIFICATION_SPLITS = os.path.join(BOXCARS_DATASET_ROOT, "classification_splits.pkl")

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2017"

############################################################
#  Configurations
############################################################


class CarsConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cars"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # COCO has 80 classes

#%%
class BoxCarsDataset(object):
    def __init__(self, load_atlas = False, load_split = None, use_estimated_3DBB = False, estimated_3DBB_path = None):
        self.dataset = load_cache(BOXCARS_DATASET)
        self.use_estimated_3DBB = use_estimated_3DBB
        
        self.atlas = None
        self.split = None
        self.split_name = None
        self.estimated_3DBB = None
        self.X = {}
        self.Y = {}
        for part in ("train", "validation", "test"):
            self.X[part] = None
            self.Y[part] = None # for labels as array of 0-1 flags
            
        if load_atlas:
            self.load_atlas()
        if load_split is not None:
            self.load_classification_split(load_split)
        if self.use_estimated_3DBB:
            self.estimated_3DBB = load_cache(estimated_3DBB_path)
        
    #%%
    def load_atlas(self):
        self.atlas = load_cache(BOXCARS_ATLAS)
    
    #%%
    def load_classification_split(self, split_name):
        self.split = load_cache(BOXCARS_CLASSIFICATION_SPLITS)[split_name]
        self.split_name = split_name
       
    #%%
    def get_image(self, vehicle_id, instance_id):
        """
        returns decoded image from atlas in RGB channel order
        """
        return cv2.cvtColor(cv2.imdecode(self.atlas[vehicle_id][instance_id], 1), cv2.COLOR_BGR2RGB)
        
    #%%
    def get_vehicle_instance_data(self, vehicle_id, instance_id, original_image_coordinates=False):
        """
        original_image_coordinates: the 3DBB coordinates are in the original image space
                                    to convert them into cropped image space, it is necessary to subtract instance["3DBB_offset"]
                                    which is done if this parameter is False. 
        """
        vehicle = self.dataset["samples"][vehicle_id]
        instance = vehicle["instances"][instance_id]
        if not self.use_estimated_3DBB:
            bb3d = self.dataset["samples"][vehicle_id]["instances"][instance_id]["3DBB"]
        else:
            bb3d = self.estimated_3DBB[vehicle_id][instance_id]
            
        if not original_image_coordinates:
            bb3d = bb3d - instance["3DBB_offset"]

        return vehicle, instance, bb3d 
            
       
    #%%
    def initialize_data(self, part):
        assert self.split is not None, "load classification split first"
        assert part in self.X, "unknown part -- use: train, validation, test"
        assert self.X[part] is None, "part %s was already initialized"%part
        data = self.split[part]
        x, y = [], []
        for vehicle_id, label in data:
            num_instances = len(self.dataset["samples"][vehicle_id]["instances"])
            x.extend([(vehicle_id, instance_id) for instance_id in range(num_instances)])
            y.extend([label]*num_instances)
        self.X[part] = np.asarray(x,dtype=int)

        y = np.asarray(y,dtype=int)
        y_categorical = np.zeros((y.shape[0], self.get_number_of_classes()))
        y_categorical[np.arange(y.shape[0]), y] = 1
        self.Y[part] = y_categorical
        


    def get_number_of_classes(self):
        return len(self.split["types_mapping"])
        
        
    def evaluate(self, probabilities, part="test", top_k=1):
        samples = self.X[part]
        assert samples.shape[0] == probabilities.shape[0]
        assert self.get_number_of_classes() == probabilities.shape[1]
        part_data = self.split[part]
        probs_inds = {}
        for vehicle_id, _ in part_data:
            probs_inds[vehicle_id] = np.zeros(len(self.dataset["samples"][vehicle_id]["instances"]), dtype=int)
        for i, (vehicle_id, instance_id) in enumerate(samples):
            probs_inds[vehicle_id][instance_id] = i
            
        get_hit = lambda probs, gt: int(gt in np.argsort(probs.flatten())[-top_k:])
        hits = []
        hits_tracks = []
        for vehicle_id, label in part_data:
            inds = probs_inds[vehicle_id]
            hits_tracks.append(get_hit(np.mean(probabilities[inds, :], axis=0), label))
            for ind in inds:
                hits.append(get_hit(probabilities[ind, :], label))
                
        return np.mean(hits), np.mean(hits_tracks)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on BoxCars.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on BoxCars")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/BoxCars/",
                        help='Directory of the BoxCars dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CarsConfig()
    else:
        class InferenceConfig(CarsConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = BoxCarsDataset()
        #-----------------here so far--------------------
        dataset_train.load_coco(args.dataset, "train", year=args.year, auto_download=args.download)
        if args.year in '2014':
            dataset_train.load_coco(args.dataset, "valminusminival", year=args.year, auto_download=args.download)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "val" if args.year in '2017' else "minival"
        dataset_val.load_coco(args.dataset, val_type, year=args.year, auto_download=args.download)
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all',
                    augmentation=augmentation)

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "val" if args.year in '2017' else "minival"
        coco = dataset_val.load_coco(args.dataset, val_type, year=args.year, return_coco=True, auto_download=args.download)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
