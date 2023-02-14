from transformers import TFRobertaModel, RobertaTokenizer
import tensorflow as tf

class Roberta ():

  def __init__(self, loss = tf.keras.losses.CategoricalCrossentropy(),  maxSentenceLen = 90, modelid = 'roberta-large',
               lr = 1e-6, metrics = ['accuracy'], num_labels = 7, patience= 3, dropout= 0.2):

    # if str.lower(modelid) == 'base':
    #   premodel = 'roberta-base'
    # elif str.lower(modelid) == 'large':
    #   premodel = 'roberta-large'
    # else:
    #   raise ValueError('The pretrained model indicated is wrong. Use "base" or "large".')
    
    self.tokenizer = RobertaTokenizer.from_pretrained(modelid, padding= 'left', truncation= 'left', max_length= maxSentenceLen) # pad and truncate left to remove things from context not from utterance 

    optimizer = tf.optimizers.Adam(lr)# try using transformers.AdamWeightDecay
    self.labels = { 0: 'neutral', 1: 'anger', 2: 'disgust', 3: 'fear', 4: 'joy', 5: 'sadness', 6: 'surprise'}
    input_ids = tf.keras.layers.Input(shape=(maxSentenceLen,), dtype=tf.int32)
    attention_mask = tf.keras.layers.Input(shape=(maxSentenceLen,), dtype=tf.int32)

    bobby = TFRobertaModel.from_pretrained(modelid, num_labels = num_labels, id2label = self.labels)(input_ids, attention_mask)
    dense = tf.keras.layers.Dense(1024)(bobby.pooler_output)
    drop = tf.keras.layers.Dropout(dropout)(dense)
    output = tf.keras.layers.Dense(num_labels, activation ='softmax')(drop)
    
    self.callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = patience, restore_best_weights=True)
    self.model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs= output)
    self.model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

    self.maxSentenceLen = maxSentenceLen

    self.args={'Sentence Length: ': str(maxSentenceLen),
               'Num Classes: ': str(num_labels), 
               'Patience: ': str(patience),
               'Dropout Rate:' : str(dropout), 
               'Learning Rate: ' : str(lr),
               'Model Type: ': modelid}

  def fit(self, X, y, val_data, epochs = 5, batch_size = 64, callbacks=None, class_weight = None):
    self.model.fit(X,y,validation_data = val_data,
                   epochs=epochs, 
                   batch_size = batch_size,
                   callbacks= callbacks, 
                   class_weight = class_weight)
    
  def predict(self,X):
    return self.model.predict(X)
  
  def evaluate(self, X, y):
    return self.model.evaluate(X,y)
  
  def summary(self):
    return self.model.summary()
  
  def save_weights (self, path):
    self.model.save_weights(path)
  
  def load_weights(self, path):
    self.model.load_weights(path)

  def save_model(self, path):
    self.model.save_model(path)
  
  def generate_input(self, utterance, context):
    return context + '</s></s>' + utterance # this is the input format for roberta

  def tokenize(self, utterance, context):
    return self.tokenizer.encode_plus(self.generate_input(utterance, context), return_tensors='tf', padding= 'max_length', truncation= 'longest_first', max_length= self.maxSentenceLen)
  
  def inference(self, utterance, context):
    tokenized_input = self.tokenize(utterance, context)
    return self.model.predict([tokenized_input['input_ids'], tokenized_input['attention_mask']])

