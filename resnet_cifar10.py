# USAGE
# python resnet_cifar10.py --checkpoints output/checkpoints
# python resnet_cifar10.py --checkpoints output/checkpoints \
# 	--model output/checkpoints/epoch_50.hdf5 --start-epoch 50

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import ResNet
from pyimagesearch.callbacks import EpochCheckpoint
from pyimagesearch.callbacks import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.models import load_model
import keras.backend as K
import numpy as np
import argparse
import sys

# set a high recursion limit so Theano doesn't complain
sys.setrecursionlimit(5000)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-c", "--checkpoints", required=True,
# 	help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str,
	help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
	help="epoch to restart training at")
args = vars(ap.parse_args())

# load the training and testing data, converting the images from
# integers to floats
# print("[INFO] loading CIFAR-10 data...")
# ((trainX, trainY), (testX, testY)) = cifar10.load_data()
# trainX = trainX.astype("float")
# testX = testX.astype("float")

# apply mean subtraction to the data
# mean = np.mean(trainX, axis=0)
# trainX -= mean
# testX -= mean

# convert the labels from integers to vectors
# lb = LabelBinarizer()
# trainY = lb.fit_transform(trainY)
# testY = lb.transform(testY)
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(config.RESIZE,config.RESIZE)
pp = PatchPreprocessor(config.RESIZE,config.RESIZE)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, aug=aug,
	preprocessors=[pp,mp, iap], classes=2)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE,
	preprocessors=[sp, mp, iap], classes=2)
testGen = HDF5DatasetGenerator(config.TEST_HDF5, config.BATCH_SIZE,
	preprocessors=[sp,mp,iap], classes=config.NUM_CLASSES)
# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")
#

# if there is no specific model checkpoint supplied, then initialize
# the network (ResNet-56) and compile the model
if args["model"] is None:
	print("[INFO] compiling model...")
	opt = SGD(lr=1e-1)
	model = ResNet.build(config.RESIZE, config.RESIZE, 3, 10, (9, 9, 9),
		(64, 64, 128, 256), reg=0.0005)
	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])

# otherwise, load the checkpoint from disk
else:
	print("[INFO] loading {}...".format(args["model"]))
	model = load_model(args["model"])

	# update the learning rate
	print("[INFO] old learning rate: {}".format(
		K.get_value(model.optimizer.lr)))
	K.set_value(model.optimizer.lr, 1e-5)
	print("[INFO] new learning rate: {}".format(
		K.get_value(model.optimizer.lr)))

# construct the set of callbacks
callbacks = [
	EpochCheckpoint(config.CHECKPOINTS, every=5,
		startAt=args["start_epoch"]),
	TrainingMonitor("output/resnet56_cifar10.png",
		jsonPath="output/resnet56_cifar10.json",
		startAt=args["start_epoch"])]

# train the network
print("[INFO] training network...")
model.fit_generator(
	trainGen.generator(),
	steps_per_epoch=trainGen.numImages//config.BATCH_SIZE,
	validation_data=valGen.generator(),
	validation_steps=valGen.numImages//config.BATCH_SIZE,
	epochs=config.EPOCHS,
	max_queue_size=10
	callbacks=callbacks, verbose=1)

trainGen.close()
valGen.close()
testGen.close()
