from bert import tokenization

folder = '/content/drive/My Drive/RuBERT'
vocab_path = folder+'/vocab.txt'

tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=False)

def tokenize1(sentence, max_seq_length):
  '''Токенизация одного предложения с маской или без.
  
  ::param::sentence - предложение строкой (str)
  ::param::max_seq_length - максимальная длина последовательности (int)
  
  ::return::token_input - массив индексов токенов длины max_seq_length, соотв. предложению sentence (np.array(int))
  ::return::mask_input - массив масок (1 - [MASK]; 0 - нет) длины max_seq_length, соотв. предложению sentence (np.array(int))
  ::return::seg_input - массив индексов предложений (0 - первое; 1 - второе) длины max_seq_length, соотв. предложению sentence (np.array(int))
  
  
  '''
  
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

  token_input = token_input + [0] * (max_seq_length - len(token_input))

  mask_input = [0] * max_seq_length
  for i in range(len(mask_input)):
    if token_input[i] == 103:
        mask_input[i] = 1

  seg_input = [0] * max_seq_length

  token_input = np.asarray([token_input])
  mask_input = np.asarray([mask_input])
  seg_input = np.asarray([seg_input])
  
  return token_input, mask_input, seg_input, tokens

def tokenize2(sentence1, sentence2, max_seq_length):
  
  tokens1 = tokenizer.tokenize(sentence1)
  tokens2 = tokenizer.tokenize(sentence2)

  tokens = ['[CLS]'] + tokens1 + ['[SEP]'] + tokens2 + ['[SEP]']
  
  token_input = tokenizer.convert_tokens_to_ids(tokens)      
  token_input = token_input + [0] * (max_seq_length - len(token_input))
  
  mask_input = [0] * max_seq_length
  
  seg_input = [0] * max_seq_length
  len_1 = len(tokens1) + 2     
  for i in range(len(tokens2) + 1):  
    seg_input[len_1 + i] = 1  

  token_input = np.asarray([token_input])
  mask_input = np.asarray([mask_input])
  seg_input = np.asarray([seg_input])
  
  return token_input, mask_input, seg_input, tokens
  
def get_word_input(tokens):
  '''Метод для 
  
  '''
  iword = 0
  word_input = list()
  word_input.append(0)

  for t in range(1, len(tokens)):
    if '##' in tokens[t]:
      word_input.append(iword)
    else:
      iword = iword + 1
      word_input.append(iword)
      
  return word_input
