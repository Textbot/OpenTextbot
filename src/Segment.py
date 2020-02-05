'''
Class and methods for working with segments.
'''

'''

'''
ListAttractorLength = 10

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
