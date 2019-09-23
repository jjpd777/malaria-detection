import matplotlib
matplotlib.use("Agg")
from utils import SimplePreprocessor
from utils import HDF5DatasetGenerator
from utils import config as config
from utils.config import store_params
from utils import ResNet
from utils import EpochCheckpoint
from utils import TrainingMonitor
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
aug = ImageDataGenerator(rescale= 1 / 255.0,rotation_range=20, zoom_range=0.05,
	width_shift_range=0.05, height_shift_range=0.05, shear_range=0.05,
	horizontal_flip=True, fill_mode="nearest")
valaug = ImageDataGenerator(rescale= 1 / 255.0)
# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, aug=aug,
	preprocessors=[sp], classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE, aug=valaug,
	preprocessors=[sp], classes=config.NUM_CLASSES)

if args["model"] is None:
	print("[INFO] compiling model...")
	opt = SGD(lr=config.LEARNING_RATE,momentum=config.MOMENTUM)
	model = ResNet.build(config.RESIZE, config.RESIZE, config.NUM_CHANNELS,
	 config.NUM_CLASSES, config.STAGES,config.FILTERS, reg=config.NETWORK_REG)
	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])
else:
	print("[INFO] loading {}...".format(args["model"]))
	model = load_model(args["model"])

	# update the learning rate
	print("[INFO] old learning rate: {}".format(
		K.get_value(model.optimizer.lr)))
	K.set_value(model.optimizer.lr,config.SECOND_LR)
	print("[INFO] new learning rate: {}".format(
		K.get_value(model.optimizer.lr)))

# construct the set of callbacks
callbacks = [
	EpochCheckpoint(config.CHECKPOINTS, every=20,
		startAt=args["start_epoch"]),
	TrainingMonitor(config.MONITOR_PATH_PNG,
		jsonPath=config.MONITOR_PATH_JSON,
		startAt=args["start_epoch"])]

# train the network
print("[INFO] training network...")
model.fit_generator(
	trainGen.generator(),
	steps_per_epoch=trainGen.numImages//config.BATCH_SIZE,
	validation_data=valGen.generator(),
	validation_steps=valGen.numImages//config.BATCH_SIZE,
	epochs=config.EPOCHS,
	max_queue_size=10,
	callbacks=callbacks, verbose=1)

trainGen.close()
valGen.close()
