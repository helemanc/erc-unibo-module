import json 
import numpy as np  

idx2emotion = { 0: 'neutral', 1: 'anger', 2: 'disgust', 3: 'fear', 4: 'joy', 5: 'sadness', 6: 'surprise'}
emotion2idx = {v:k for k,v in idx2emotion.items()}


def generate_output(emotion_prediction, output_path, modelid, utterance, context):
    """
    Generate the output for the model

    Parameters
    ----------
    emotion_prediction : str
        The emotion prediction

    Returns
    -------
    output : json
        The output of the model
    """
    output = {
              'model_used': modelid,
              'context': context,
              'utterance': utterance,
              }

    output['emotion_prediction'] = {k:str(np.round(v,7)) for k,v in zip(emotion2idx.keys(), emotion_prediction[0])}
    # save json as file
    print(output)
    with open(output_path, 'w') as f:
        json.dump(output, f)
