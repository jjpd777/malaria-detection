import json
# define the paths to the images directory
IMAGES_PATH = "./clean_data/train/"

RESIZE = 64 
NUM_CLASSES = 2
DATASET_MEAN = "./output/malaria_mean.json"

EPOCHS = 50
BATCH_SIZE = 64 
LEARNING_RATE = 0.01
MOMENTUM = 0
DECAY = LEARNING_RATE/EPOCHS
## IF FINE TUNING FCHEAD
FCH1 = 256
FCH2 = 128

TRAIN_HDF5 = "./clean_data/hdf5/train.hdf5"
VAL_HDF5 = "./clean_data/hdf5/val.hdf5"
TEST_HDF5 = "./clean_data/hdf5/test.hdf5"
PARAMS = "parameters.txt"

EXPERIMENT_NAME = "./output/experiment-1/"
PARAMS_FILE = EXPERIMENT_NAME + PARAMS
MODEL_PATH = EXPERIMENT_NAME + "finetune.model"
OUTPUT_PATH = EXPERIMENT_NAME 

def store_params():
    data= {}
    data['hyperparameters'] = []
    data['hyperparameters'].append({
        'image_size' : RESIZE,
        'epochs' : EPOCHS,
        'batch_size' : BATCH_SIZE,
        'learning_rate' : LEARNING_RATE,
        'momentum' : LEARNING_RATE,
        'decay' : DECAY,
        'fc_layer_1' : FCH1,
        'fc_layer_2' : FCH2
        })
    with open(PARAMS_FILE,'w') as write:
        json.dump(data,write)


