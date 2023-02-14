from setfit import SetFitModel, SetFitTrainer, SetFitHead
from sentence_transformers.losses import CosineSimilarityLoss
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from torch import Union, nn
from torch.jit.annotations import Optional,Tuple, Dict
from sklearn.linear_model import LogisticRegression

class SetFit(SetFitModel):
    def __init__(self,  modelid= 'all-roberta-large-v1', kerashead= True) -> None:
        super(SetFitModel,self).__init__()
        
        self.model = SetFitModel.from_pretrained('sentence-transformers/'+ modelid, use_differentiable_head= True, head_params={'out_features': 7})
        self.embedding_dimension = self.model.model_body[1].get_sentence_embedding_dimension()
        if kerashead:
            self.model_head = SetFitClassifier(lr_head = 1e-4, patience = 5, embedding_dimension = self.embedding_dimension)
        self.modelid = modelid
    
    def inference(self, utterance, context):
        embedded_utterance = self.model.model_body.encode(utterance)
        embedded_context = self.model.model_body.encode(context)
        
        return self.model.model_head.predict([embedded_context, embedded_utterance])
    

class SetFitClassifier():
    def __init__(self, lr_head, patience, embedding_dimension, concat = True):
        layers = tf.keras.layers
        contextIn = layers.Input((embedding_dimension,),name ='ContextInputHead')
        utteranceIn = layers.Input((embedding_dimension,),name= 'UtteranceInputHead')
        
        if concat:
            concat = layers.Concatenate(name = 'Concatenation')([contextIn, utteranceIn])
            # dense1 = layers.Dense(embedding_dimension, activation= 'relu')(utteranceIn)
            mergeLayer = concat
        else:
            #Attention Layer definition
            attention = layers.Attention()([utteranceIn, contextIn])
            # weightContext = tf.matmul(utteranceIn, attention, transpose_b = True)
            # mul = layers.Multiply()([utteranceIn, attention])
            drop = layers.Dropout(0.1)(attention)
            # addnorm = layers.LayerNormalization()(utteranceIn + drop)
            # # norm= layers.LayerNormalization()(attention + utteranceIn)
            # dense1 = layers.Dense(embedding_dimension, activation = 'relu', name = 'DenseFromConcat')(addnorm)
            # addnorm2 = layers.LayerNormalization()(dense1 + addnorm)
            concat = layers.Concatenate()([utteranceIn, drop])
            
            mergeLayer = drop   
        
        dense2 = layers.Dense(1024, activation = 'relu', name = 'Dense')(mergeLayer)
        # drop = layers.Dropout(0.15)(dense1)
        output = layers.Dense(7, activation = 'softmax', name = 'Output')(dense2)
        optimizer = tf.keras.optimizers.Adam(learning_rate= lr_head)
        loss = tf.keras.losses.CategoricalCrossentropy()
        self.callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = patience)
        self.classHead = tf.keras.Model(inputs= [contextIn,utteranceIn], outputs= output)

        self.classHead.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])


    def fit (self, input, labels, val_input, val_labels, batch_size,epochs = 30):
        self.classHead.fit(x = input,
                              y = to_categorical(labels),
                              batch_size = batch_size,
                              validation_data = (val_input, to_categorical(val_labels)),
                              epochs = epochs,
                              callbacks = self.callback)

    def summary(self):
        return self.classHead.summary()

    def predict(self,X):
        return self.classHead.predict(X)
    
    def save_weights (self, path):
        self.classHead.save_weights(path)
  
    def load_weights(self, path):
        self.classHead.load_weights(path) 

    def save_model(self, path):
        self.classHead.save_model(path) #Non so come vengano gestiti i pesi in torch