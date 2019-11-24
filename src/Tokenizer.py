from bert import tokenization

folder = '/content/drive/My Drive/RuBERT'
vocab_path = folder+'/vocab.txt'

tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=False)

def tokenize1(sentence):
  
  sentence = sentence.replace(' [MASK] ','[MASK]'); sentence = sentence.replace('[MASK] ','[MASK]'); sentence = sentence.replace(' [MASK]','[MASK]')  # удаляем лишние пробелы
  sentence = sentence.split('[MASK]')             # разбиваем строку по маске
  tokens = ['[CLS]']                              # фраза всегда должна начинаться на [CLS]
  # обычные строки преобразуем в токены с помощью tokenizer.tokenize(), вставляя между ними [MASK]
  for i in range(len(sentence)):
    if i == 0:
        tokens = tokens + tokenizer.tokenize(sentence[i]) 
    else:
        tokens = tokens + ['[MASK]'] + tokenizer.tokenize(sentence[i]) 
  tokens = tokens + ['[SEP]']

  token_input = tokenizer.convert_tokens_to_ids(tokens)

  token_input = token_input + [0] * (512 - len(token_input))

  mask_input = [0]*512
  for i in range(len(mask_input)):
    if token_input[i] == 103:
        mask_input[i] = 1

  seg_input = [0]*512

  token_input = np.asarray([token_input])
  mask_input = np.asarray([mask_input])
  seg_input = np.asarray([seg_input])
  
  return token_input, mask_input, seg_input, tokens

def tokenize2(sentence1, sentence2):
  
  tokens1 = tokenizer.tokenize(sentence1)
  tokens2 = tokenizer.tokenize(sentence2)

  tokens = ['[CLS]'] + tokens1 + ['[SEP]'] + tokens2 + ['[SEP]']
  
  token_input = tokenizer.convert_tokens_to_ids(tokens)      
  token_input = token_input + [0] * (512 - len(token_input))
  
  mask_input = [0] * 512
  
  seg_input = [0]*512
  len_1 = len(tokens1) + 2                   # длина первой фразы, +2 - включая начальный CLS и разделитель SEP
  for i in range(len(tokens2)+1):            # +1, т.к. включая последний SEP
    seg_input[len_1 + i] = 1                # маскируем вторую фразу, включая последний SEP, единицами

  token_input = np.asarray([token_input])
  mask_input = np.asarray([mask_input])
  seg_input = np.asarray([seg_input])
  
  return token_input, mask_input, seg_input, tokens
  
def get_word_input(tokens):
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
