'''
Class and methods for working with segments.
'''

import OpenTextbot.src.Algebra as Algebra
import OpenTextbot.src.Tokenizer as Tokenizer

'''
Hyperparameters
'''
ListAttractorLength = 10
max_seq_length = 100

tokens_bpemb = ['▁ма', 'ма', '▁мы', 'ла', '▁ра', 'му']

def get_word_input_bpemb(tokens_bpemb, max_seq_length):
  '''Метод для генерации массива индексов слов из токенов BPEmb.
  :param tokens: массив токенов
  :param max_seq_length: максимальная длина последовательности (int)
  
  :return word_input: массив индексов слов длины max_seq_length, соотв. предложению sentence (np.array(int)) 
  '''
  iword = 1
  word_input = list()
  word_input.append(iword)

  for t in range(1, len(tokens)):
    if '▁' not in tokens[t]:
      word_input.append(iword)
    else:
      iword = iword + 1
      word_input.append(iword)
  
  word_input = word_input + [0] * (max_seq_length - len(word_input))
  
  return word_input

word_input = get_word_input(tokens, 20)
print(word_input)

class Segment(object):
  
  def __init__(self, ListTE, ListTokenID):
    self.ListTE = ListTE
    self.ListTokenID = ListTokenID
  
  def mean(self):
    '''
    1). 1 среднее значение
    2). среднее значение+ нули, длина равно len(ListTE)
    3). ListAttractor
    '''
