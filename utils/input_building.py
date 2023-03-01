import json
from utils.text_processing import *
import models
# from models.emoberta import model
#from emoberta import model

# variables 

# MODELS_MAPPING = {
#     'emoberta': model.Roberta,
#     #'setfit': models.setfit.model.SetFit,
# }


def input_processing(model, json_file, current_utterance_field  = 'user-turn', context_utterance_field = 'agent-turn'):
    """
    Process the input and return the model input
    
    Parameters
    ----------
    model : str
        The model name
    json_file : json
        The .json file

        
    """

    current_utterance, context_utterance = read_json(json_file, current_utterance_field, context_utterance_field)
    proc_current_utterance = preprocessing_pipeline(current_utterance)
    proc_context_utterance = preprocessing_pipeline(context_utterance)
    #model_input = generate_input(proc_current_utterance, proc_context_utterance, model)
    #return model_input

    return proc_current_utterance, proc_context_utterance
    



def read_json(json_file, user_turn_field, agent_turn_field): 
    """
    Read the json file and return the current utterance and the context utterance

    Parameters
    ----------
    json_file : str
        The path to the json file
    user_turn_field : str
        The field name of the user turn
    agent_turn_field : str
        The field name of the agent turn

    Returns
    -------
    current_utterance : str
        The current utterance
    context_utterance : str
        The context utterance
    """

    with open(json_file) as f:
        data = json.load(f)
        current_utterance = data['items'][-1][user_turn_field]
        context_utterance = data['items'][-1][agent_turn_field]
    return current_utterance, context_utterance

'''
def generate_input(proc_current_utterance, proc_context_utterance, model):
    """
    Generate the input for the model

    Parameters
    ----------
    proc_current_utterance : str
        The processed current utterance
    proc_context_utterance : str
        The processed context utterance
    model : str
        The model name

    Returns
    -------
    model_input_string : str
        The model input
    """

    MODEL_MAPPING_GENERATION = {
        'emoberta': lambda x, y: MODELS_MAPPING[model].generate_input(x, y),
        #'setfit': lambda x, y: MODELS_MAPPING[model].generate_input(x, y),
    }

    if model in MODEL_MAPPING_GENERATION:
        model_input = MODEL_MAPPING_GENERATION[model](proc_current_utterance, proc_context_utterance)
    else:
        raise ValueError("Model not supported")

    return model_input 
'''    



