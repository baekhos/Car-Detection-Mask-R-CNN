

import skimage.io
import pandas as pd
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import skimage.draw
import pickle
import matplotlib.pyplot as plt
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import datetime
import os
import cv2
#%%
# Root directory of the project
#Paths to dataset
BOXCARS_DATASET_ROOT = "D:\\Master TAID\\Anul2\\MLAV\\Car-Detection-Mask-R-CNN\\DataSet\\BoxCars116k"

#%%

BOXCARS_IMAGES_ROOT = os.path.join(BOXCARS_DATASET_ROOT, "images")
BOXCARS_DATASET = os.path.join(BOXCARS_DATASET_ROOT, "dataset.pkl")
BOXCARS_ATLAS = os.path.join(BOXCARS_DATASET_ROOT, "atlas.pkl")
BOXCARS_CLASSIFICATION_SPLITS = os.path.join(BOXCARS_DATASET_ROOT, "classification_splits.pkl")


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

    def initialize_data(self, part):
        self.X = {}
        self.Y = {}
        self.part=part
        self.add_class("car", 1, "car")
        for part in ("train", "validation", "test"):
            self.X[part] = None
            self.Y[part] = None  # for labels as array of 0-1 flags
        self.dataset = self.load_cache(BOXCARS_DATASET)
        self.atlas = self.load_cache(BOXCARS_ATLAS)
        self.split = self.load_cache(BOXCARS_CLASSIFICATION_SPLITS)['hard']
        self.nr_of_classes = len(self.split["types_mapping"])
        self.df = pd.read_pickle(BOXCARS_DATASET)
        assert self.split is not None, "load classification split first"
        assert part in self.X, "unknown part -- use: train, validation, test"
        assert self.X[part] is None, "part %s was already initialized" % part
        data = self.split[self.part]
        x, y = [], []
        for vehicle_id, label in data:
            num_instances = len(self.dataset["samples"][vehicle_id]["instances"])
            x.extend([(vehicle_id, instance_id) for instance_id in range(num_instances)])
            y.extend([label] * num_instances)
        self.X[self.part] = np.asarray(x, dtype=int)
        for x in self.X[self.part]:
            vehicle_id, instance_id = x
            image = self.get_image_by_id(vehicle_id, instance_id)
            height, width = image.shape[:2]
            image_path=self.df['samples'][vehicle_id]['instances'][instance_id]['path']
            _, filename = os.path.split(image_path)
            image_path=os.path.join(BOXCARS_IMAGES_ROOT,image_path)
            self.add_image(
                "car",
                image_id=filename,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=1)

    def load_cache(self,path, encoding="latin-1", fix_imports=True):
        with open(path, "rb") as f:
            return pickle.load(f, encoding=encoding, fix_imports=True)

    def get_image(self, image_id):
        """
        returns decoded image from atlas in RGB channel order
        """
        vehicle_id, instance_id = self.X[self.part][image_id]
        return cv2.cvtColor(cv2.imdecode(self.atlas[vehicle_id][instance_id], 1), cv2.COLOR_BGR2RGB)

    def get_image_by_id(self, vehicle_id, instance_id):
        """
        returns decoded image from atlas in RGB channel order
        """
        return cv2.cvtColor(cv2.imdecode(self.atlas[vehicle_id][instance_id], 1), cv2.COLOR_BGR2RGB)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "car":
            return super(self.__class__, self).load_mask(image_id)
        image = self.get_image( image_id)
        height, width = image.shape[:2]
        mask = np.zeros([height, width, 1], dtype=np.uint8)
        start, end = self.getMask2D( image_id)
        rr, cc = skimage.draw.rectangle(start, end, shape=image.shape[:2])
        mask[rr, cc] = 1
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def getMask3D(self,  index):
        vehicle_id, instance_id = self.X[self.part][index]
        points = np.array(self.df['samples'][vehicle_id]['instances'][instance_id]['3DBB'], np.int32)
        X = points[:, 0]
        Y = points[:, 1]
        return X, Y, np.array(points)

    def getMask2D(self,  index):
        vehicle_id, instance_id = self.X[self.part][index]
        x1, y1, x2, y2 = self.df['samples'][vehicle_id]['instances'][instance_id]['2DBB']
        return (int(y1), int(x1)), (int(y2 + y1), int(x2 + x1))

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "car":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)





def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CarsDataset()
    dataset_train.initialize_data("train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CarsDataset()
    dataset_val.initialize_data("validation")
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

config = CarsConfig()
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax
