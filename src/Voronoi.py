#!/usr/bin/env python3
'''
Кластеризация пространства ВП для быстрого поиска по сжатым ВП средствами ассиметричных ячеек Вороного.
'''

import io

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np

import Algebra

'''
Класс для представления ассиметричных ячеек Вороного.
Атрибуты:
    ArrayCentroids - массив центроидов кластеров Вороного;
    ListBoolSubCluster - список указателей на то, имеет ли кластер подкластеры: если 0, то не имеет, а если не ноль то это глобальный индекс кластера;
    ListClusterListPoints - список индексов точек из ArrayCompressedWE, входящих в кластер при условии, что ListBoolSubCluster[i] == 0. В противном случае имеем глобальный индекс кластера;
    ListArrayCentroids - список массивов центороидов вложенных кластеров (подкластеров) Вороного;
    ListListClusterListPoints - список индексов точек из ArrayCompressedWE, входящих в подкластеры.
'''
class Voronoi (object):
    def __init__(self, ArrayCentroids, ListBoolSubCluster, ListClusterListPoints, ListArrayCentroids, ListListClusterListPoints):
        self.ArrayCentroids = ArrayCentroids
        self.ListBoolSubCluster = ListBoolSubCluster
        self.ListClusterListPoints = ListClusterListPoints
        self.ListArrayCentroids = ListArrayCentroids
        self.ListListClusterListPoints = ListListClusterListPoints

'''
Метод получения массива центроидов для ячеек Вороного:
'''
def GetCentroids(ArrayWE, ListClusterSize):
    
    kmeans = KMeans(n_clusters=ListClusterSize, random_state=0).fit(ArrayWE)
    ArrayCentroids = kmeans.cluster_centers_
    
    return ArrayCentroids

'''
Метод быстрого получения массива центроидов для ячеек Вороного:
'''
def GetCentroidsFast(ArrayWE, ListClusterSize, BatchSize):
    
    kmeans = MiniBatchKMeans(n_clusters=ListClusterSize, random_state=0, batch_size=BatchSize).fit(ArrayWE)
    ArrayCentroids = kmeans.cluster_centers_
    
    return ArrayCentroids

'''
Метод записи массива центроидов для ячеек Вороного в текстовый файл.

Вход:
Filename_Ac (str) - путь к файлу;
ArrayCentroids (np.array(np.float32)) - массив центроидов кластеров Вороного.

Выход:
Нет.
'''
def ExportCentroids(Filename_Ac, ArrayCentroids):
    
    f = open(Filename_Ac, 'a', encoding='utf-8-sig')    
    for i in range(len(ArrayCentroids)):
        for j in ArrayCentroids[i]:
            f.write(str(j) + ' ')    
        f.write('\n') 
    
    return 0

'''
Метод чтения массива центроидов для ячеек Вороного из текстового файла.

Вход:
Filename_Ac (str) - путь к файлу;
EmbeddingSize (int) - размерность пространства ВП.

Выход:
ArrayCentroids (np.array(np.float32)) - массив центроидов кластеров Вороного.
'''
def ImportCentroids(Filename_Ac, EmbeddingSize):
    
    Reader_Ac = io.open(Filename_Ac, 'r', encoding='utf-8-sig', newline='\n', errors='ignore')
    ListCentroids = list()
    for line in Reader_Ac: 
        tokens = line.rstrip().split(' ')
        X = np.array(tokens, dtype=np.float32)
        X1 = X.reshape((EmbeddingSize)) 
        ListCentroids.append(X1)    
    ArrayCentroids = np.asarray(ListCentroids, np.float32)
    
    return ArrayCentroids


'''
Метод кластеризации сжатых ВП для быстрого поиска по областям k-мерного пространства.
Вход:
    ArrayWE - массив ВП для генерации кластеров верхнего уровня;
    ArrayCompressedWE - массив сжатых ВП;
    ListClusterSize - число кластеров (подкластеров в кластере);
    EmbeddingSize - размерность пространства ВП;
    GlobalArrayCentroids - массив центроидов для ArrayCompressedWE.
Выход:
    ModelVoronoi(Voronoi) - модель ассиметричных ячеек Вороного:
        ModelVoronoi.ArrayCentroids - массив центроидов кластеров Вороного;
        ModelVoronoi.ListBoolSubCluster - список указателей на то, имеет ли кластер подкластеры: если 0, то не имеет, а если не ноль то это глобальный индекс кластера;
        ModelVoronoi.ListClusterListPoints - список индексов точек из ArrayCompressedWE, входящих в кластер при условии, что ListBoolSubCluster[i] == 0. В противном случае имеем глобальный индекс кластера;
        ModelVoronoi.ListArrayCentroids - список массивов центороидов вложенных кластеров (подкластеров) Вороного;
        ModelVoronoi.ListListClusterListPoints - список индексов точек из ArrayCompressedWE, входящих в подкластеры.
''' 
def VoronoiClustering(ArrayWE, ArrayCompressedWE, ListClusterSize, EmbeddingSize, GlobalArrayCentroids):
    ...
    return ArrayCentroids, ListBoolSubCluster, ListClusterListPoints, ListArrayCentroids, ListListClusterListPoints 

