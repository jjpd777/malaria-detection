import json
import os
IMAGES_PATH = "./clean_data/train/"
TRAIN_HDF5 = "./clean_data/hdf5/train.hdf5"
VAL_HDF5 = "./clean_data/hdf5/val.hdf5"
TEST_HDF5 = "./clean_data/hdf5/test.hdf5"
HDF5_FILES = [TRAIN_HDF5,VAL_HDF5,TEST_HDF5]

RESIZE = 64
NUM_CHANNELS = 3
NUM_CLASSES = 2


EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.1
SECOND_LR = 0.005
MOMENTUM = 0
NETWORK_REG = 0.0005
DECAY = LEARNING_RATE/EPOCHS
STAGES = (3,4,6)
FILTERS = (128,64,128,256)

FINE_TUNE_BOOL = False
FCH1 = 0
FCH2 = 0


EXPERIMENT_NAME = "./output/experiment-2/"
CHECKPOINTS = EXPERIMENT_NAME + "checkpoints/"
MONITOR_PATH_PNG = EXPERIMENT_NAME + "monitor.png"
MONITOR_PATH_JSON = EXPERIMENT_NAME + "monitor.json"
PARAMS = "parameters.txt"
PARAMS_FILE = EXPERIMENT_NAME + PARAMS
LOG_NAME = EXPERIMENT_NAME + "console.log"
MODEL_PATH = EXPERIMENT_NAME + "resnet.model"
OUTPUT_PATH = EXPERIMENT_NAME

def make_experiment():
    os.mkdir(EXPERIMENT_NAME)
    os.mkdir(CHECKPOINTS)
    os.mknod(LOG_NAME)

def store_params():
    data= {}
    data['hyperparameters'] = []
    data['hyperparameters'].append({
        'image_size' : RESIZE,
        'epochs' : EPOCHS,
        'batch_size' : BATCH_SIZE,
        'learning_rate' : LEARNING_RATE,
        'second_lr': SECOND_LR,
        'stages': STAGES,
        'filters': FILTERS,
        'momentum' : MOMENTUM,
        'fine_tune' :{
            'fine_tune_used': FINE_TUNE_BOOL,
            'fc_layer_1' : FCH1,
            'fc_layer_2' : FCH2}
        })
    with open(PARAMS_FILE,'a') as write:
        json.dump(data,write)
