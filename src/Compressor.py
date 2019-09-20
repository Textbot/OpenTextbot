#!/usr/bin/env python3

'''
Methods for word vectors compression and working with compressed word vectors.

Методы сжатия ВП и работы с квантованными векторами.
'''

import io

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np

import Algebra

'''
Класс для хранения сжатых моделей ВП.

Атрибуты:
    ArrayCompressedWE (np.array(np.uint8)) - массив сжатых ВП;
    ArrayCentroids (np.array(np.float32)) - 3-мерный массив [ListSubvectorSize, ListClusterSize, SubvectorSize], содержащий координаты центроидов кластеров.
    
'''

class ClassCompressedWE (object):
    def __init__(self, ArrayCompressedWE, ArrayCentroids):
        self.ArrayCompressedWE = ArrayCompressedWE
        self.ArrayCentroids = ArrayCentroids

'''
Метод записи сжатых ВП в текстовый файл.

Вход:
    Filename (str) - путь к файлу;
    ArrayCompressedWE (np.array(np.uint8)) - массив сжатых ВП.
Выход:
    Нет.
'''
def ExportCompressedWE(Filename, ArrayCompressedWE):
  
    f = open(Filename, 'a', encoding='utf-8-sig')
    for i in range(len(ArrayCompressedWE)):
        for j in ArrayCompressedWE[i]:
            f.write(str(j) + ' ')    
    f.write('\n')    
   
  return 0
  

'''
Метод импорта сжатых ВП.

Вход:
    Filename (str) - путь к файлу.
Выход:
    ArrayCompressedWE (np.array(np.uint8)) - массив сжатых ВП.
'''
def ImportCompressedWE(Filename):

    Reader = io.open(Filename, 'r', encoding='utf-8-sig', newline='\n', errors='ignore')
    ListCompressedWE = list()
    for line in Reader: 
        tokens = line.rstrip().split(' ')        
        X = np.array(tokens, dtype=np.uint8)       
        ListCompressedWE.append(X)    
    ArrayCompressedWE = np.asarray(ListCompressedWE, np.uint8)
    
    return ArrayCompressedWE


'''
Метод, записывающий координаты центроидов кластеров в текстовый файл.

Вход:
    Filename (str) - путь к файлу;
    ArrayCentroinds (np.array(np.float32)) - 3-мерный массив [ListSubvectorSize, ListClusterSize, SubvectorSize],
        содержащий координаты центроидов кластеров;
    ListSubvectorSize (int) - число подвекторов, например, 75, если 300-мерный вектор разбивается на 4-мерные подвектора;
    ListClusterSize (int) - число кластеров для каждого подвектора, например, 256, т.к. удобно хранить в uint8 / byte;
    SubvectorSize (int) - размерность подвектора, например 4.
Выход:
    Нет.
'''
def ExportCentroids(Filename, ArrayCentroinds, ListSubvectorSize, ListClusterSize, SubvectorSize):
    
    f = open(Filename, 'a', encoding='utf-8-sig') 
    for k in range(ListSubvectorSize):
        for j in range(ListClusterSize):
            for i in range(SubvectorSize):
                f.write(str(ArrayCentroinds[i, j, k]) + ' ')    
        f.write('\n')    
    
    return 0


'''
Метод, считывающий координаты центроидов кластеров из текстового файла.

Вход:
    Filename (str) - путь к файлу;
    ListClusterSize (int) - число кластеров для каждого подвектора, например, 256, т.к. удобно хранить в uint8 / byte;
    SubvectorSize (int) - размерность подвектора, например 4.
Выход:
    ArrayCentroids (np.array(np.float32)) - 3-мерный массив [ListSubvectorSize, ListClusterSize, SubvectorSize],
        содержащий координаты центроидов кластеров.
    
'''
def ImportCentroids(Filename, ListClusterSize, SubvectorSize):
    
    Reader = io.open(Filename, 'r', encoding='utf-8-sig', newline='\n', errors='ignore')
    ListCentroids = list()
    for line in Reader: 
        tokens = line.rstrip().split(' ') #1024
        #Reshape to 256 x 4
        X = np.array(tokens, dtype=np.float32)
        X1 = X.reshape((ListClusterSize, SubvectorSize))       
        ListCentroids.append(X1)   
    ArrayCentroids = np.asarray(ListCentroids, np.float32)
    
    return ArrayCentroids


'''
Метод сжатия ВП.
Вход:
    ArrayWE (np.array(np.float32)) - массив векторных представлений;
    VocabularySize (int) - словарный запас, которым мы хотим оперировать, например, 100000 словоформ;
    EmbeddingSize (int) - размерность пространства ВП;
    ListClusterSize (int) - число кластеров для каждого подвектора, например, 256, т.к. удобно хранить в uint8 / byte;
    SubvectorSize (int) - размерность подвектора, например 4;
    BatchSize (int) - размер выборки для поиска кластеров методом MiniBatchKMeans, например, 1000.
Выход:
    ArrayCentroids (np.array(np.float32)) - 3-мерный массив [ListSubvectorSize, ListClusterSize, SubvectorSize],
        содержащий координаты центроидов кластеров;
    ArrayCompressedWE (np.array(np.uint8)) - массив сжатых ВП.
'''
def CompressWE(ArrayWE, VocabularySize, EmbeddingSize, ListClusterSize, SubvectorSize, BatchSize):
    
    ListSubvectorSize = int(EmbeddingSize / SubvectorSize)
    X1 = ArrayWE.reshape((VocabularySize, ListSubvectorSize, SubvectorSize))
    ListCentroids = list()
    ListCompressorWEs = list()
    for i in range(ListSubvectorSize):
        x = X1[:, i, :]
        kmeans = MiniBatchKMeans(n_clusters=ListClusterSize, random_state=0, batch_size=BatchSize).fit(x)
        ArrayCentroid = kmeans.cluster_centers_
        ListCentroids.append(ArrayCentroid)
        ArrayPredict = kmeans.predict(x).astype(np.uint8)
        ListCompressorWEs.append(ArrayPredict)        
    ArrayCentroids = np.asarray(ListCentroids, np.float32).transpose()
    ArrayCompressedWE = np.asarray(ListCompressorWEs, np.uint8).transpose()
    
    return ArrayCentroids, ArrayCompressedWE

'''
Метод сжатия ВП в ситуации, когда центроиды кластеров заранее известны.
Вход:
    ArrayWE (np.array(np.float32)) - массив векторных представлений;
    VocabularySize (int) - словарный запас, которым мы хотим оперировать, 
        например, 100000 словоформ;
    EmbeddingSize (int) - размерность пространства ВП;
    ListClusterSize (int) - число кластеров для каждого подвектора, например, 256, т.к. удобно хранить в uint8 / byte;
    SubvectorSize (int) - размерность подвектора, например 4;
    BatchSize (int) - размер выборки для поиска кластеров методом MiniBatchKMeans,
        например, 1000;
    ArrayCentroids (np.array(np.float32)) - 3-мерный массив [ListSubvectorSize, ListClusterSize, SubvectorSize],
        содержащий координаты центроидов кластеров.
Выход:
    ArrayCompressedWE (np.array(np.uint8)) - массив сжатых ВП.
'''
def CompressWEwithClusters(ArrayWE, VocabularySize, EmbeddingSize, ListClusterSize, SubvectorSize, BatchSize, ArrayCentroids):
    
    ListSubvectorSize = int(EmbeddingSize / SubvectorSize)
    X1 = ArrayWE.reshape((VocabularySize, ListSubvectorSize, SubvectorSize))
    ListCompressedWE = list()
    for i in range(ListSubvectorSize):
        x = X1[:, i, :]        
        kmeans = MiniBatchKMeans(n_clusters=ListClusterSize, random_state=0, batch_size=BatchSize)
        kmeans.cluster_centers_ = ArrayCentroids[i]        
        ArrayPredict = kmeans.predict(x).astype(np.uint8)
        ListCompressedWE.append(ArrayPredict)        
    ArrayCompressedWE = np.asarray(ListCompressedWE, np.uint8).transpose()
    
    return ArrayCompressedWE

'''
Метод разжатия ВП. Сжатую модель ВП всегда можно привести к состоянию,
близкому к исходному, применив данный метод.
Вход:
    ArrayCompressedWE (np.array(np.uint8)) - массив сжатых ВП;
    ArrayCentroids (np.array(np.float32)) - 3-мерный массив [ListSubvectorSize, ListClusterSize, SubvectorSize],
        содержащий координаты центроидов кластеров;
    EmbeddingSize (int) - размерность пространства ВП.
Выход:
    ArrayWE (np.array(np.float32)) - массив векторных представлений.
'''
def DecompressWE(ArrayCompressedWE, ArrayCentroids, EmbeddingSize):
    ...
    return ArrayWE

'''
Метод разжатия одного ВП. В отличие от разжатия списка ВП в DecompressWE.
Вход:
    CompressedWE (np.uint8) - сжатое ВП;
    ArrayCentroids (np.array(np.float32)) - 3-мерный массив [ListSubvectorSize, ListClusterSize, SubvectorSize],
        содержащий координаты центроидов кластеров;
    EmbeddingSize (int) - размерность пространства ВП.
Выход:
    WE (np.array(np.float32)) - векторные представление.
'''
def DecompressSingleWE(CompressedWE, ArrayCentroids, EmbeddingSize):
    ...
    return WE


'''
Метод кластеризации сжатых ВП для быстрого поиска по областям k-мерного пространства.
Вход:
    ArrayWE - массив ВП для генерации кластеров верхнего уровня;
    ArrayCompressedWE - массив сжатых ВП;
    ListClusterSize - число кластеров (подкластеров в кластере);
    EmbeddingSize - размерность пространства ВП;
    GlobalArrayCentroids - массив центроидов для ArrayCompressedWE.
Выход:
    ArrayCentroids - массив центроидов кластеров Вороного;
    ListBoolSubCluster - список указателей на то, имеет ли кластер подкластеры: если 0, то не имеет, а если не ноль то это глобальный индекс кластера;
    ListClusterListPoints - список индексов точек из ArrayCompressedWE, входящих в кластер при условии, что ListBoolSubCluster[i] == 0. В противном случае имеем глобальный индекс кластера;
    ListArrayCentroids - список массивов центороидов вложенных кластеров (подкластеров) Вороного;
    ListListClusterListPoints - список индексов точек из ArrayCompressedWE, входящих в подкластеры.
''' 
def VoronoiClustering(ArrayWE, ArrayCompressedWE, ListClusterSize, EmbeddingSize, GlobalArrayCentroids):
    ...
    return ArrayCentroids, ListBoolSubCluster, ListClusterListPoints, ListArrayCentroids, ListListClusterListPoints 

 