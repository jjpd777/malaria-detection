import matplotlib
matplotlib.use("Agg")
from utils import SimplePreprocessor
from utils import HDF5DatasetGenerator
from utils import config as config
from utils.config import store_params
from utils import ResNet
from utils import EpochCheckpoint
from utils import TrainingMonitor
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import load_model
import keras.backend as K
import numpy as np
import argparse

store_params()
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,
        help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
        help="epoch to restart training at")
args = vars(ap.parse_args())


sp = SimplePreprocessor(config.RESIZE,config.RESIZE)
testGen = HDF5DatasetGenerator(config.TEST_HDF5, config.BATCH_SIZE,
        preprocessors=[sp], classes=config.NUM_CLASSES)
print("[INFO] loading {}...".format(args["model"]))
model = load_model(args["model"])
totalTest= len(testGen.db["labels"])
predIdxs = model.predict_generator(testGen.generator(),
	steps=(totalTest // config.BATCH_SIZE)+1 )
 
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
 
# show a nicely formatted classification report
labels = list(testGen.db["labels"])
print(classification_report(labels, predIdxs))	
