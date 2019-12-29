#1. Import libraries:

!pip install keras-bert
!pip install bert-tensorflow

import sys
import codecs

import numpy as np

from bert import tokenization
from keras_bert import load_trained_model_from_checkpoint
from keras.models import Model
from keras import layers
from keras.layers import Input, Dense, BatchNormalization
from keras_pos_embd import PositionEmbedding



import OpenTextbot.src.Algebra as Algebra
import OpenTextbot.src.Compressor as Compressor
import OpenTextbot.src.Tokenizer as Tokenizer
import OpenTextbot.src.Voronoi as Voronoi


#2. Import compressed token vectors and their centroids: 

folder = '/content/drive/My Drive/RuBERT'

Filename = folder+'/ArrayCentroids.txt'
ArrayCentroidsImported = Compressor.ImportCentroids(Filename, ListClusterSize=256, SubvectorSize=4)

Filename2 = folder+'/ArrayCompressedWE.txt'
ArrayCompressedWEImported = Compressor.ImportCompressedWE(Filename2)


#3. Method for creating word vector from token vectors for one sentence:

def GetWordWE(sentence):
  
  sentence = sentence.replace(' [MASK] ','[MASK]')
  sentence = sentence.replace('[MASK] ','[MASK]') 
  sentence = sentence.replace(' [MASK]','[MASK]')
  sentence = sentence.split('[MASK]')            
  tokens = ['[CLS]']  
  
  for i in range(len(sentence)):
    if i == 0:
        tokens = tokens + tokenizer.tokenize(sentence[i]) 
    else:
        tokens = tokens + ['[MASK]'] + tokenizer.tokenize(sentence[i]) 
  tokens = tokens + ['[SEP]']

  token_input = tokenizer.convert_tokens_to_ids(tokens)

  ListCompressedWE = list()
  for i in range(len(token_input)):
    ListCompressedWE.append(ArrayCompressedWEImported[token_input[i]])
  
  ArrayCompressedWE = np.asarray(ListCompressedWE)

  ArrayWE = Compressor.DecompressListWE(ArrayCompressedWE, ArrayCentroidsImported, 768)
  
  ListWordWE = list()

  ListListWE = list()
  CurrentListWE = list()

  for t in range(0, len(tokens)):
    if '##' in tokens[t]:
      CurrentListWE.append(ArrayWE[t])
      if (t == (len(tokens) - 1)):
        ListListWE.append(CurrentListWE)
        ListWordWE.append(Algebra.Mean(CurrentListWE, Type='Cluster'))
    else:
      if(len(CurrentListWE) == 0):
        CurrentListWE.append(ArrayWE[t])        
      else:
        ListListWE.append(CurrentListWE)
        ListWordWE.append(Algebra.Mean(CurrentListWE, Type='Cluster'))
        CurrentListWE.clear()
        CurrentListWE.append(ArrayWE[t])

  
  ArrayWordWE = np.asarray(ListWordWE)
  
  Length = len(ListWordWE)

  Array768 = np.zeros(768)
  for i in range(512-len(ListWordWE)):
    ListWordWE.append(Array768)
  ArrayWordWE = np.asarray(ListWordWE)
  ArrayWordWE = ArrayWordWE.reshape([1, 512, 768])

  return ArrayWordWE, Length
 
 
#4. Create Keras-bert model for WordBERT (you can make the same with TF-2, TF-Keras or PyTorch):

folder = '/content/drive/My Drive/RuBERT'
config_path = folder + '/bert_config.json'
checkpoint_path = folder + '/bert_model.ckpt'
vocab_path = folder + '/vocab.txt'

tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=False)

model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True)

ListLayer = list()

a = Input(shape=(512,768), name='InputEmbedding')
ListLayer.append(a)

ListLayer.append(model.layers[5](ListLayer[len(ListLayer) - 1])) #PositionEmbedding
ListLayer.append(model.layers[6](ListLayer[len(ListLayer) - 1])) #Dropout
ListLayer.append(model.layers[7](ListLayer[len(ListLayer) - 1])) #LayerNormalization
for i in range(0, 12):
  ListLayer.append(model.layers[8 + 8*i](ListLayer[len(ListLayer) - 1]))
  ListLayer.append(model.layers[9 + 8*i](ListLayer[len(ListLayer) - 1]))
  ListLayer.append(model.layers[10 + 8*i]([ListLayer[len(ListLayer) - 3], ListLayer[len(ListLayer) - 1]]))
  ListLayer.append(model.layers[11 + 8*i](ListLayer[len(ListLayer) - 1]))
  ListLayer.append(model.layers[12 + 8*i](ListLayer[len(ListLayer) - 1]))
  ListLayer.append(model.layers[13 + 8*i](ListLayer[len(ListLayer) - 1]))
  ListLayer.append(model.layers[14 + 8*i]([ListLayer[len(ListLayer) - 3], ListLayer[len(ListLayer) - 1]]))
  ListLayer.append(model.layers[15 + 8*i](ListLayer[len(ListLayer) - 1]))
  
ListLayer.append(model.layers[104](ListLayer[len(ListLayer) - 1]))
ListLayer.append(model.layers[105](ListLayer[len(ListLayer) - 1]))

WordModel = Model(inputs=a, outputs=ListLayer[len(ListLayer) - 1])
for layer in WordModel.layers:
  layer.trainable = False
