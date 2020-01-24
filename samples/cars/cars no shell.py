# -*- coding: utf-8 -*-

import skimage.io
import sys
import pandas as pd
import json
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import skimage.draw
import pickle
import os
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import datetime
import os
import cv2
#%%
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
#Paths to dataset
BOXCARS_DATASET_ROOT = "C:\\Users\\George\\Desktop\\BoxCars-master\\DataBase\\BoxCars116k"

#%%

BOXCARS_IMAGES_ROOT = os.path.join(BOXCARS_DATASET_ROOT, "images")
BOXCARS_DATASET = os.path.join(BOXCARS_DATASET_ROOT, "dataset.pkl")
BOXCARS_ATLAS = os.path.join(BOXCARS_DATASET_ROOT, "atlas.pkl")
BOXCARS_CLASSIFICATION_SPLITS = os.path.join(BOXCARS_DATASET_ROOT, "classification_splits.pkl")
estimated_3DBB_path="C:\\Users\\George\\Desktop\\BoxCars-master\\data\\estimated_3DBB.pkl"
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class CarsConfig(Config):
    """Configuration for training on the car  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "car"

    # We use a GPU with 6GB memory, which can fit one image.
    # Adjust up if you use a stronger GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + car

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

class CarsDataset(utils.Dataset):



    def load_car(self,  subset):
        """Load a subset of the Car dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.X = {}
        self.Y = {}
        for part in ("train", "validation", "test"):
            self.X[part] = None
            self.Y[part] = None  # for labels as array of 0-1 flags
        self.dataset = self.load_cache(BOXCARS_DATASET)
        self.atlas = self.load_cache(BOXCARS_ATLAS)
        self.split = self.load_cache(BOXCARS_CLASSIFICATION_SPLITS)['hard']
        self.estimated_3DBB = self.load_cache(estimated_3DBB_path)
        self.nr_of_classes = len(self.split["types_mapping"])

        self.add_class("car", 1, "car")

        # Train or validation dataset?
        assert subset in ["train", "validation"]


        self.df = pd.read_pickle(BOXCARS_DATASET)
        # Add images
        for x in self.X[subset]:
            vehicle_id, instance_id = x
            vehicle, instance, polygons = self.get_vehicle_instance_data(vehicle_id, instance_id)
            image = self.get_image(vehicle_id, instance_id)
            height, width = image.shape[:2]
            image_path=self.df['samples'][vehicle_id]['instances'][instance_id]['path']
            _, filename = os.path.split(image_path)
            self.add_image(
                "car",
                image_id=filename,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def getMask2D(self, part, index):
        vehicle_id, instance_id = self.X[part][index]
        x1, y1, x2, y2 = self.df['samples'][vehicle_id]['instances'][instance_id]['2DBB']
        return (int(y1), int(x1)), (int(y2 + y1), int(x2 + x1))

    def load_mask(self, part, image_id):
        vehicle_id, instance_id = self.X[part][image_id]
        image = self.get_image(part, image_id)
        height, width = image.shape[:2]
        mask = np.zeros([height, width, 1], dtype=np.uint8)
        start, end = self.getMask2D(part, image_id)
        rr, cc = skimage.draw.rectangle(start, end, shape=image.shape[:2])
        mask[rr, cc] = 1
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "car":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def load_cache(path, encoding="latin-1", fix_imports=True):
        with open(path, "rb") as f:
            return pickle.load(f, encoding=encoding, fix_imports=True)



    def get_image(self, part,image_id):
        """
        returns decoded image from atlas in RGB channel order
        """
        vehicle_id, instance_id=self.X[part][image_id]
        return cv2.cvtColor(cv2.imdecode(self.atlas[vehicle_id][instance_id], 1), cv2.COLOR_BGR2RGB)


    def initialize_data(self, part):
        assert self.split is not None, "load classification split first"
        assert part in self.X, "unknown part -- use: train, validation, test"
        assert self.X[part] is None, "part %s was already initialized" % part
        data = self.split[part]
        x, y = [], []
        for vehicle_id, label in data:
            num_instances = len(self.dataset["samples"][vehicle_id]["instances"])
            x.extend([(vehicle_id, instance_id) for instance_id in range(num_instances)])
            y.extend([label] * num_instances)
        self.X[part] = np.asarray(x, dtype=int)

        y = np.asarray(y, dtype=int)
        y_categorical = np.zeros((y.shape[0], self.nr_of_classes))
        y_categorical[np.arange(y.shape[0]), y] = 1
        self.Y[part] = y_categorical


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CarsDataset()
    dataset_train.load_car("train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CarsDataset()
    dataset_val.load_balloon(args.dataset, "validation")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect cars.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/cars/dataset/",
                        help='Directory of the Cars dataset')
    parser.add_argument('--weights', required=True,
                        metavar="D:\\Master TAID\\Anul2\\MLAV\\Car-Detection-Mask-R-CNN\\mask_rcnn_coco.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color test effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color test effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "test":
        assert args.image or args.video,\
               "Provide --image or --video to apply color test"

    print("Weights: ", args.weights)
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
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "test":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'test'".format(args.command))
