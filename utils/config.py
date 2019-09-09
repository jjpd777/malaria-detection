# define the paths to the images directory
IMAGES_PATH = "./clean_data/train/"

RESIZE = 227 
NUM_CLASSES = 2
DATASET_MEAN = "./output/malaria_mean.json"

EPOCHS = 50
BATCH_SIZE = 64 
LEARNING_RATE = 0.02
MOMENTUM = 0
DECAY = LEARNING_RATE/EPOCHS

TRAIN_HDF5 = "./clean_data/hdf5/train.hdf5"
VAL_HDF5 = "./clean_data/hdf5/val.hdf5"
TEST_HDF5 = "./clean_data/hdf5/test.hdf5"
PARAMS = "parameters.txt"

EXPERIMENT_NAME = "./output/experiment-1/"
PARAMS_FILE = EXPERIMENT_NAME + PARAMS
MODEL_PATH = EXPERIMENT_NAME + "finetune.model"
OUTPUT_PATH = EXPERIMENT_NAME 


