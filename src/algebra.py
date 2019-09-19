"""Algebra for Language Models."""

#[АКТУАЛЬНО - 19.09.2019]

import math
import numpy as np
from scipy.spatial.distance import cdist, euclidean


'''
Метод нахождения классической евклидовой метрики в пространстве ВП 

    Вход:
        ListWE (list(np.array(np.float32))) - список векторных представлений,
            среди которых мы будем искать наиболее близкую точку к WE;
        WE (np.array(np.float32)) - исходное векторное представление.
        
    Выход:
        ListDistance - список мер близости WE к каждой точке из ListWE.
        
'''
def Euclidian(ListWE, WE):  
    
  ListDistance = cdist(ListWE, [WE])

  return ListDistance

'''
Метод нахождения наиболее близкой точки по классической евклидовой метрике в пространстве ВП 

    Вход:
        ListWE (list(np.array(np.float32))) - список векторных представлений,
            среди которых мы будем искать наиболее близкую точку к WE;
        WE (np.array(np.float32)) - исходное векторное представление.
        
    Выход: 
        ID (int) - индекс точки из ListWE, наиболее близкой к WE.
        
'''
def EuclidianMax(ListWE, WE):
    
  ListDistance = cdist(ListWE, [WE])
  ID = np.argmin(ListDistance)
    
  return ID

'''
Метод нахождения n наиболее близких точек по классической евклидовой метрике в пространстве ВП 

    Вход:
        CurrentListWE (list(np.array(np.float32))) - список векторных представлений,
            среди которых мы будем искать наиболее близкую точку к WE;
        CurrentWE (np.array(np.float32)) - исходное векторное представление;
        N (int) - число точек.
        
    Выход: 
        CurrentListID (list(int)) - список индексов точек из ListWE, наиболее близких к WE.
        
'''
def EuclidianMaxN(CurrentListWE, CurrentWE, N):
    
    ListDistance = cdist(CurrentListWE, [CurrentWE])
    CurrentArrayID = np.argsort(ListDistance.flatten())[0:N]
    CurrentListID = list(CurrentArrayID)
    
    return CurrentListID

'''
Скалярное произведение ВП. 
Может использоваться как косинусная метрика для нормализованных (!) векторов.

Вход:
    WE1 (np.array(np.float32)) - векторное представление 1;
    WE2 (np.array(np.float32)) - векторное представление 2.
    
Выход: 
    DP (np.float32) - скалярное произведение ВП1 и ВП2.
'''
def DotProduct(WE1, WE2):
  
  WE1 = np.asarray(WE1, dtype=np.float32)
  WE2 = np.asarray(WE2, dtype=np.float32)
  DP = np.dot(WE1, WE2)

  return DP

'''
Метод вычисления нормы (длины) вектора

Вход:
    WE (np.array) - векторное представление;
    EmbeddingSize (int) - размерность пространства ВП.
    
Выход: 
    Norm (float) - норма вектора (длина радиус-вектора)
'''
def GetVectorNorm(WE, EmbeddingSize):
  
  Norm = 0.0
  for i in range(EmbeddingSize):
    Norm = Norm + WE[i]*WE[i]
    Norm = math.sqrt(Norm)

  return Norm

'''
Метод сложения векторов в пространстве ВП.

Вход:
    WE1 (np.array) - векторное представление 1;
    WE2 (np.array) - векторное представление 2.
    
Выход: 
    WE3 (np.array) - ВП3, полученное в результате сложения ВП1 и ВП2.
'''
def Add(WE1, WE2):
    
  for i in range(len(WE1)):
        WE1[i] = WE1[i] + WE2[i]
    WE3 = WE1
    
  return WE3

'''
Метод сложения произвольного числа векторов в пространстве ВП.

Вход:
    ListWE (list(np.array)) - список векторных представлений;
    EmbeddingSize (int) - размерность пространства ВП.
    
Выход: 
    WE (np.array) - ВП, полученное в результате сложения ВП из списка ListWE.
'''
def Addn(ListWE, EmbeddingSize):
  ...
  return WE

'''
Метод вычисления разности векторов в пространстве ВП.

Вход:
    WE1 (np.array) - векторное представление 1;
    WE2 (np.array) - векторное представление 2.
    
Выход: 
    WE3 (np.array) - ВП3, полученное в результате разности ВП1 и ВП2.
'''
def Subtract(WE1, WE2):
  ...
  return WE3
