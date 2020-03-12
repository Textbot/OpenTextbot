#!/usr/bin/env python3

"""Algebra for Language Models."""


import math
import numpy as np
from scipy.spatial.distance import cdist, euclidean
from sklearn.cluster import KMeans



def Euclidian(ListWE, WE): 
    '''Метод нахождения классической евклидовой метрики в пространстве ВП 

    :param ListWE: список векторных представлений (list(np.array(np.float32))) 
    :param WE: векторное представление (np.array(np.float32))
        
    :return ListDistance: список мер близости WE к каждой точке из ListWE    
    '''    
    ListDistance = cdist(ListWE, [WE])

    return ListDistance


def EuclidianMax(ListWE, WE):
    '''Метод нахождения наиболее близкой точки по классической евклидовой метрике в пространстве ВП 

    :param ListWE: список векторных представлений (list(np.array(np.float32))) 
    :param WE: векторное представление (np.array(np.float32))
        
    :return ID: индекс точки из ListWE, наиболее близкой к WE  (int)
    '''
    ListDistance = cdist(ListWE, [WE])
    ID = np.argmin(ListDistance)
    
    return ID


def EuclidianMaxN(ListWE, WE, N):
    '''Метод нахождения n наиболее близких точек по классической евклидовой метрике в пространстве ВП 

    :param ListWE: список векторных представлений (list(np.array(np.float32))) 
    :param WE: векторное представление (np.array(np.float32))
    :param N: число точек (int)
        
    :return ListID: список индексов точек из ListWE, наиболее близких к WE  (list(int))        
    '''
    ListDistance = cdist(ListWE, [WE])
    ArrayID = np.argsort(ListDistance.flatten())[0:N]
    ListID = list(ArrayID)
    
    return ListID


def DotProduct(WE1, WE2):
    '''Скалярное произведение ВП. Может использоваться как косинусная метрика для нормализованных (!) векторов.

    :param WE1: векторное представление 1 (np.array(np.float32))
    :param WE2: векторное представление 2 (np.array(np.float32))
    
    :return DP: скалярное произведение ВП1 и ВП2 (np.float32)
    '''
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


def Addn(ListWE, EmbeddingSize):
    '''Метод сложения произвольного числа векторов в пространстве ВП.

    :param ListWE: список векторных представлений (list(np.array)).
    :param EmbeddingSize: размерность пространства ВП (int).    
    :return WE: ВП, полученное в результате сложения ВП из списка ListWE (np.array).
    '''
    WEn = np.zeros(EmbeddingSize, dtype=float)
    for i in range(len(ListWE)):
        WEn = Add(WEn, ListWE[i])
    WE = WEn.astype(np.float32)

    return WE

    
def Subtract(WE1, WE2):
    '''Метод вычисления разности векторов в пространстве ВП.

    :param WE1: векторное представление 1 (np.array).
    :param WE2: векторное представление 2 (np.array).    
    :return WE3: векторное представление 3, полученное в результате разности ВП1 и ВП2 (np.array).
    '''
    WE2 = WE2 * (-1)
    for i in range(len(WE1)):
        WE1[i] = WE1[i] + WE2[i]
    WE3 = WE1

    return WE3


def Mean(ListWE, Type=None):
  '''Метод поиска "средних" значений списка ВП.
    
  :param ListWE: список векторных представлений (list(np.array(np.float32)))
  :param Type: тип "среднего" ('Cluster' - центроид кластера, 'GM' - геометрическая медиана, ['Mean', None] - np.mean)
  :return WE: "среднее" значение ВП (np.array(np.float32))
   
  '''
  ArrayWE = np.asarray(ListWE, dtype=np.float32)
  WE = np.zeros(len(ListWE[0]))
    
  if (len(ListWE) == 1):
    WE = ListWE[0]
  else:
    if Type == 'Cluster':
        kmeans = KMeans(n_clusters=1, random_state=0).fit(ArrayWE)
        CentroidWE = kmeans.cluster_centers_
        WE = np.array(CentroidWE[0])
    elif Type == 'GM':
        eps=1e-5 #Максимальная разница между двумя прогнозируемыми точками
        ArrayWE64 = ArrayWE.astype(np.float64)
        X = ArrayWE64
    
        y = np.mean(X, 0)

        while True:
            D = cdist(X, [y])
            nonzeros = (D != 0)[:, 0]

            Dinv = 1 / D[nonzeros]
            Dinvs = np.sum(Dinv)
            W = Dinv / Dinvs
            T = np.sum(W * X[nonzeros], 0)

            num_zeros = len(X) - np.sum(nonzeros)
            if num_zeros == 0:
                y1 = T
            elif num_zeros == len(X):
                GM = y.astype(np.float32)
                WE = GM
                break
            else:
                R = (T - y) * Dinvs
                r = np.linalg.norm(R)
                rinv = 0 if r == 0 else num_zeros/r
                y1 = max(0, 1-rinv)*T + min(1, rinv)*y

            if euclidean(y, y1) < eps:
                GM = y1.astype(np.float32)
                WE = GM
                break

            y = y1
    else: #'Mean' or None
        WE = np.mean(ArrayWE, axis=0)
    
  return WE
