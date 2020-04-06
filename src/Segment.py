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
  '''Класс для представления сегмента
  :param Text: текст сегмента
  :param ListTE: список векторных представлений токенов. Смещается в процессе работы сети
  :param ListTokenID: список индексов токенов
  :param ListWE: список векторных представлений слов. Смещается в процессе работы сети
  :param ListWordID: список индексов слов. Словарь формируется при чтении CONLL
  :param SE: векторное представление сегмента. Смещается в процессе работы сети
  '''
  def __init__(self, Text=None, ListTE=None, ListTokenID=None, ListWE=None, ListWordID=None, SE=None, ListVertexID = None):
    self.Text = Text
    self.ListTE = ListTE
    self.ListWE = ListWE
    self.ListTokenID = ListTokenID
    self.ListWordID = ListWordID
    self.SE = SE  
    self.ListVertexID = ListVertexID

  def conll(self, ListCONLL, Vocabulary):
    ListTE = list()
    ListTokenID = list()
    ListWE = list()
    ListWordID = list()
    Text = ''
    for token in ListCONLL:
      word = token.form
      Text = Text + ' ' + word
      head = token.head
      wordID = 0
      if word not in Vocabulary:
        Vocabulary.append(word)
        wordID = len(Vocabulary)
      else:
        for v in range(len(Vocabulary)):
          if Vocabulary[v] == word:
            wordID = v
      ListWordID.append(wordID)
      #Токенезируем word:
      ListWordTokenID = bpemb_ru.encode_ids(word)
      ListTokenID.extend(ListWordTokenID)
      
      ListWordTE = bpemb_ru.embed(word)
      ListTE.extend(ListWordTE)
      ListWE.append(Algebra.Mean(ListWordTE))
    #self.Text = ListCONLL.text
    self.Text = Text
    self.ListTE = ListTE
    self.ListWE = ListWE
    self.ListTokenID = ListTokenID
    self.ListWordID = ListWordID
    self.SE = Algebra.Mean(ListTE)
  
    
