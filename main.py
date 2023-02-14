from utils import input_building, output_building, download_weights
import json 
# from models.emoberta import model
# from models.setfit import model
from models.emoberta import emoberta 
#from models.setfit import setfit
import numpy as np
import os
import transformers

# set tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
transformers.logging.CRITICAL


# read configuration file
with open("config.json") as f:
    data = json.load(f)
print("Configuration file read")

MODEL = data['model_name']
MODELID = data['pretained_model_name']
JSON_INPUT_FILE = data['json_input_file_path']
JSON_OUTPUT_FILE = data['json_output_file_path']
URL_WEIGHTS = data['url_weights_' + MODEL]

MODEL_MAP = { 'emoberta': emoberta.Roberta#,
              #'setfit': setfit.SetFit
            }

print("Model: ", MODEL)

# build model input
print("Building model input...")
proc_utterance, proc_context = input_building.input_processing(MODEL, JSON_INPUT_FILE)
print("Model input built")

# download weights
# if weights are not present in the models folder, download them
weights_path = 'models/' +  MODEL + '/weights.h5'
if not os.path.exists(weights_path):
    download_weights.download(URL_WEIGHTS, MODEL)

# builds model
print("Building model...")
if MODEL in MODEL_MAP.keys():
    model = MODEL_MAP[MODEL](modelid = MODELID)
    model.load_weights(weights_path)
else:
    raise ValueError('Model not found')
print("Model built")

# generate prediction 
print("Generating prediction...")
emotion_prediction = model.inference(proc_utterance, proc_context)
print(emotion_prediction)
print("Prediction generated")

# generate output 
print("Generating output...")
output = output_building.generate_output(emotion_prediction, JSON_OUTPUT_FILE, MODELID, proc_utterance, proc_context)
print("Output generated")


