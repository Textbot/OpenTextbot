TestWords = ['kettle', 'fly', 'bed']
for Word in TestWords:
  TestWE = Algebra.Mean(bpemb_en.embed('kettle'))
  ListWE = [ObjectEmbedding, ProcessEmbedding]
  ListD = ['Object', 'Process']
  TWE = Algebra.EuclidianMax(ListWE, TestWE)
  print('\''+ Word + '\' is the ' + ListD[TWE] + '.')
