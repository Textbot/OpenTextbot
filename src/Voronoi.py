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

'''

'''
def VoronoiClusteringWithCentroids(ArrayCompressedWE, GlobalArrayCentroids, EmbeddingSize, Parameter, ArrayCentroids, ListClusterListPoints, ListClusterSize):
    
    ListBoolSubCluster = list()
    
    CurrentClusterIndex = int(len(ArrayCompressedWE))
    
    #3. Если в кластере более 1000 точек при ListClusterSize == 100, то
    #его необходимо снова кластеризовать. Для это индексы его точек
    #нужно заменить на индекс самого кластера (CurrentClusterIndex).
    #Значения CurrentClusterIndex начинаются с 1000000 и т.д.
    for i in ListClusterListPoints:
        if (len(i) > int(ListClusterSize * Parameter)):
            ListBoolSubCluster.append(CurrentClusterIndex)
            #i.clear()
            #i.append(CurrentClusterIndex)
            CurrentClusterIndex = CurrentClusterIndex + 1            
        else:
            ListBoolSubCluster.append(0)
    
    ListArrayCentroids1 = list()
    
    ListListClusterListPoints1 = list() #10000 списков. Экспортировать как 10000 строк, ID точек делить пробелом.
        
    for i in range(len(ListClusterListPoints)):
        if (ListBoolSubCluster[i] > 0):
            CurrentListDecompressedWE = list()
            #Восстановим каждую точку из кластера и добавим в список.
            for j in ListClusterListPoints[i]:
                CurrentListDecompressedWE.append(DecompressSingleWE(ArrayCompressedWE[j], GlobalArrayCentroids, EmbeddingSize))
            #Превратим список в массив и подадим на вход 
            CurrentArrayDecompressedWE = np.asarray(CurrentListDecompressedWE, np.float32)
            #Найдем 100 центроидов и разделим все точки на 100 кластеров:
            kmeans1 = KMeans(n_clusters=ListClusterSize, random_state=0).fit(CurrentArrayDecompressedWE)
            ArrayCentroids1 = kmeans1.cluster_centers_
            ListArrayCentroids1.append(ArrayCentroids1) # 100x100=10000
            ArrayCompressedWE1 = kmeans1.predict(CurrentArrayDecompressedWE).astype(np.int32)
        
            ListClusterListPoints1 = [[] for i in range(ListClusterSize)]
        
            for j in range (len(ArrayCompressedWE1)):
                SubClusterID = ArrayCompressedWE1[j]
                WeIDi = ListClusterListPoints[i]
                WeID = WeIDi[j]
                
                ListClusterListPoints1[SubClusterID].append(WeID)
            
            ListListClusterListPoints1.append(ListClusterListPoints1)
    

    for i in range(len(ListClusterListPoints)):
        if (ListBoolSubCluster[i] > 0):
            ListClusterListPoints[i].clear()
            ListClusterListPoints[i].append(ListBoolSubCluster[i])
    
    return ListBoolSubCluster, ListClusterListPoints, ListArrayCentroids1, ListListClusterListPoints1

