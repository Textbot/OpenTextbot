'''
Кластеризация пространства ВП для быстрого поиска по сжатым ВП средствами ассиметричных ячеек Вороного.
'''

import io

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np

import OpenTextbot.src.Algebra as Algebra
import OpenTextbot.src.Compressor as Compressor

'''
Класс для представления ассиметричных ячеек Вороного.
Атрибуты:
    ArrayCentroids - массив центроидов кластеров Вороного;
    ListSubCluster - список указателей на то, имеет ли кластер подкластеры: если 0, то не имеет, а если не ноль то это глобальный индекс кластера;
    ListClusterListPoints - список индексов точек из ArrayWE, входящих в кластер при условии, что ListSubCluster[i] == 0. В противном случае имеем глобальный индекс кластера;
    ListArrayCentroids - список массивов центороидов вложенных кластеров (подкластеров) Вороного;
    ListListClusterListPoints - список индексов точек из ArrayWE, входящих в подкластеры.
'''
class Voronoi (object):
    def __init__(self, ArrayCentroids, ListSubCluster, ListClusterListPoints, ListArrayCentroids, ListListClusterListPoints):
        self.ArrayCentroids = ArrayCentroids
        self.ListSubCluster = ListSubCluster
        self.ListClusterListPoints = ListClusterListPoints
        self.ListArrayCentroids = ListArrayCentroids
        self.ListListClusterListPoints = ListListClusterListPoints

'''
Метод получения массива центроидов для ячеек Вороного:
Вход:
ArrayWE (np.array(np.float32)) - массив ВП токенов;
ListClusterSize (int) - число кластеров Вороного 1-го уровня.

Выход:
ArrayCentroids (np.array(np.array(np.float32))) - массив центроидов кластеров Вороного.
'''
def GetCentroids(ArrayWE, ListClusterSize):
    
    kmeans = KMeans(n_clusters=ListClusterSize, random_state=0).fit(ArrayWE)
    ArrayCentroids = kmeans.cluster_centers_
    
    return ArrayCentroids

'''
Метод быстрого получения массива центроидов для ячеек Вороного:
Вход:
ArrayWE (np.array(np.float32)) - массив ВП токенов;
ListClusterSize (int) - число кластеров Вороного 1-го уровня;
BatchSize (int) - размер выборки.

Выход:
ArrayCentroids (np.array(np.array(np.float32))) - массив центроидов кластеров Вороного.
'''
def GetCentroidsFast(ArrayWE, ListClusterSize, BatchSize):
    
    kmeans = MiniBatchKMeans(n_clusters=ListClusterSize, random_state=0, batch_size=BatchSize).fit(ArrayWE)
    ArrayCentroids = kmeans.cluster_centers_
    
    return ArrayCentroids

'''
Метод записи массива центроидов для ячеек Вороного в текстовый файл.

Вход:
Filename_Ac (str) - путь к файлу;
ArrayCentroids (np.array(np.array(np.float32))) - массив центроидов кластеров Вороного.

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
ArrayCentroids (np.array(np.array(np.float32))) - массив центроидов кластеров Вороного.
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
Метод генерации кластеров Вороного 1-го уровня на основе центроидов Вороного.
Вход:
ArrayWE (np.array(np.array(np.float32))) - массив ВП токенов;
ArrayCentroids (np.array(np.array(np.float32))) - массив центроидов кластеров Вороного;
ListClusterSize (int) - число кластеров Вороного 1-го уровня;
BatchSize (int) - размер выборки.

Выход:
ListClusterListPoints (List(List(int))) - список индексов ВП, входящих в кластер, при условии, что ListSubCluster[i] == 0. 
                        В противном случае имеем глобальный индекс кластера.

'''
def GetClustersVoronoi(ArrayWE, ArrayCentroids, ListClusterSize):
    
    kmeans = MiniBatchKMeans(n_clusters=ListClusterSize, random_state=0, batch_size=100)
    kmeans.cluster_centers_ = ArrayCentroids
    
    ListClusterListPoints = [[] for i in range(ListClusterSize)]
    
    ArrayClusteredWE = kmeans.predict(ArrayWE).astype(np.int32)
    for i in range (len(ArrayWE)):
        ID = ArrayClusteredWE[i]
        ListClusterListPoints[ID].append(i)

    return ListClusterListPoints


'''
...
Метод кластеризации сжатых ВП для быстрого поиска по областям k-мерного пространства.
Вход:
    ArrayWE (np.array(np.array(np.float32))) - массив ВП токенов;
    ListClusterSize (int) - число кластеров (подкластеров в кластере);
    EmbeddingSize (int) - размерность пространства ВП;
    ArrayCentroids (np.array(np.array(np.float32))) - массив центроидов кластеров Вороного.
        
Выход:
    ModelVoronoi(Voronoi) - модель ассиметричных ячеек Вороного:
        ModelVoronoi.ArrayCentroids - массив центроидов кластеров Вороного;
        ModelVoronoi.ListSubCluster - список указателей на то, имеет ли кластер подкластеры: если 0, то не имеет, а если не ноль то это глобальный индекс кластера;
        ModelVoronoi.ListClusterListPoints - список индексов точек из ArrayWE, входящих в кластер при условии, что ListSubCluster[i] == 0. В противном случае имеем глобальный индекс кластера;
        ModelVoronoi.ListArrayCentroids - список массивов центороидов вложенных кластеров (подкластеров) Вороного;
        ModelVoronoi.ListClusterListPoints (List(List(int))) - список индексов ВП, входящих в кластер, при условии, 
                                что ListSubCluster[i] == 0. В противном случае имеем глобальный индекс кластера.
'''
def VoronoiClustering(ArrayWE, ListClusterSize, EmbeddingSize, ArrayCentroids):
    
    ListClusterListPoints = GetClustersVoronoi(ArrayWE, ArrayCentroids, ListClusterSize)
    
    ListSubCluster = list()
    
    CurrentClusterIndex = int(len(ArrayWE))
    
    #Если в кластере более 1000 точек при ListClusterSize == 100, то
    #его необходимо снова кластеризовать. Для это индексы его точек
    #нужно заменить на индекс самого кластера (CurrentClusterIndex).
    #Например, при мощности словаря в 1000000 значений, значения CurrentClusterIndex начинаются с 1000000 и т.д.
    for i in ListClusterListPoints:
        if (len(i) > int(ListClusterSize * 10)):
            ListSubCluster.append(CurrentClusterIndex)
            CurrentClusterIndex = CurrentClusterIndex + 1            
        else:
            ListSubCluster.append(0)
    
    ListArrayCentroids1 = list()
    
    ListListClusterListPoints1 = list() #10000 списков. Экспортировать как 10000 строк, ID точек делить пробелом.
        
    for i in range(len(ListClusterListPoints)):
        if (ListSubCluster[i] > 0):
            CurrentListWE = list()
            #Восстановим каждую точку из кластера и добавим в список.
            for j in ListClusterListPoints[i]:
                CurrentListWE.append(ArrayWE[j])
            #Превратим список в массив и подадим на вход 
            CurrentArrayWE = np.asarray(CurrentListWE, np.float32)
            #Найдем 100 центроидов и разделим все точки на 100 кластеров:
            kmeans1 = KMeans(n_clusters=ListClusterSize, random_state=0).fit(CurrentArrayWE)
            ArrayCentroids1 = kmeans1.cluster_centers_
            ListArrayCentroids1.append(ArrayCentroids1) # 100x100=10000
            ArrayPoints1 = kmeans1.predict(CurrentArrayWE).astype(np.int32)
        
            ListClusterListPoints1 = [[] for i in range(ListClusterSize)]
        
            for j in range (len(ArrayPoints1)):
                SubClusterID = ArrayPoints1[j]
                WeIDi = ListClusterListPoints[i]
                WeID = WeIDi[j]
                
                ListClusterListPoints1[SubClusterID].append(WeID)
            
            ListListClusterListPoints1.append(ListClusterListPoints1)
    

    for i in range(len(ListClusterListPoints)):
        if (ListSubCluster[i] > 0):
            ListClusterListPoints[i].clear()
            ListClusterListPoints[i].append(ListSubCluster[i])
    
    ModelVoronoi = Voronoi(ArrayCentroids, ListSubCluster, ListClusterListPoints, ListArrayCentroids1, ListListClusterListPoints1)
    
    return ModelVoronoi


'''
Метод записи модели Вороного в текстовые файлы.
Вход:
    Filename_Ac (str) - путь к файлу с ArrayCentroids;
    Filename_LSC (str) - путь к файлу с ListSubCluster;
    Filename_LCLP (str) - путь к файлу с ListClusterListPoints;
    Filename_Lac (str) - путь к файлу с ListArrayCentroids;
    Filename_Llclp (str) - путь к файлу с ListListClusterListPoints;
    ArrayCentroids (array(array(np.float32))) - массив центроидов кластеров 1-го уровня;
    ListSubCluster (int) - список индексов подкластеров кластора;
    ListClusterListPoints (List(List(int))) - список индексов точек из ArrayWE, входящих в кластер при условии, 
                                              что ListSubCluster[i] == 0. В противном случае имеем глобальный индекс кластера;
    ListArrayCentroids (List(array(array(np.float32)))) - массив центроидов кластеров 2-го уровня;
    ListListClusterListPoints (List(List(List(int)))) - список индексов точек из ArrayWE, входящих в подкластеры.
Выход:
Нет.
'''
def VoronoiExport(Filename_Ac, Filename_LBSC, Filename_LCLP,
                   Filename_Lac, Filename_Llclp, 
                   ArrayCentroids, ListSubCluster, ListClusterListPoints, 
                   ListArrayCentroids, ListListClusterListPoints):
    
    #1. Запишем ВП центроидов 2-го уровня из ArrayCentroids в Filename_Ac:
    #Аналогично записи любых ВП, сжимать смысла нет, т.к. размер мал,
    #а используется в разжатом виде.
    f01 = open(Filename_Ac, 'a', encoding='utf-8-sig')
    
    for i in range(len(ArrayCentroids)):
        for j in ArrayCentroids[i]:
            f01.write(str(j) + ' ')    
        f01.write('\n') 
    f01.close()
    
    #2. Запишем список ListSubCluster:
    f02 = open(Filename_LBSC, 'a', encoding='utf-8-sig')
    
    for k in ListSubCluster:
        f02.write(str(k))    
        f02.write('\n')
    f02.close()
    
    #3. Запишем списки индексов словоформ первого этапа кластеризации:
    f03 = open(Filename_LCLP, 'a', encoding='utf-8-sig')
    
    for j in range (len(ListClusterListPoints)):
        for k in ListClusterListPoints[j]:
            f03.write(str(k) + ' ')    
        f03.write('\n')
    f03.close()
    
    #4. Запишем ВП центроидов 1-го уровня из ListArrayCentroids1 в Filename_Lac:
    f04 = open(Filename_Lac, 'a', encoding='utf-8-sig')
    
    for i in range(len(ListArrayCentroids)):
        for j in (ListArrayCentroids[i]):
            for k in j:
                f04.write(str(k) + ' ')    
            f04.write('\n') 
    f04.close()
    
    #5. Запишем списки индексов словоформ (один список - одна строка):
    f05 = open(Filename_Llclp, 'a', encoding='utf-8-sig')
    
    for i in range(len(ListListClusterListPoints)):
        for j in (ListListClusterListPoints[i]):
            for k in j:
                f05.write(str(k) + ' ')    
            f05.write('\n') 
    f05.close()
    
    return 0


'''
Метод импорта модели Вороного из текстовых файлов.
Вход:
    Filename_Ac (str) - путь к файлу с ArrayCentroids;
    Filename_LSC (str) - путь к файлу с ListSubCluster;
    Filename_LCLP (str) - путь к файлу с ListClusterListPoints;
    Filename_Lac (str) - путь к файлу с ListArrayCentroids1;
    Filename_Llclp (str) - путь к файлу с ListListClusterListPoints1;
    ListClusterSize (int) - мощность множества кластеров 1-го уровня;
    EmbeddingSize (int) - размерность пространства ВП.
Выход:
    ArrayCentroids (array(array(np.float32))) - массив центроидов кластеров 1-го уровня;
    ListSubCluster (List(int)) - список индексов подкластеров кластера;
    ListClusterListPoints (List(List(int))) - список индексов точек из ArrayWE, входящих в кластер при условии, 
                                              что ListSubCluster[i] == 0. В противном случае имеем глобальный индекс кластера;
    ListArrayCentroids (List(array(array(np.float32)))) - массив центроидов кластеров 2-го уровня;
    ListListClusterListPoints (List(List(List(int)))) - список индексов точек из ArrayWE, входящих в подкластеры.
'''

def VoronoiImport(Filename_Ac, Filename_LSC, Filename_LCLP,
                   Filename_Lac, Filename_Llclp, ListClusterSize, EmbeddingSize):
    #1. Filename_Ac:
    Reader_Ac = io.open(Filename_Ac, 'r', encoding='utf-8-sig', newline='\n', errors='ignore')
    ListCentroids = list()
    for line in Reader_Ac: 
        tokens = line.rstrip().split(' ')
        X = np.array(tokens, dtype=np.float32)
        X1 = X.reshape((EmbeddingSize))
        
        ListCentroids.append(X1)
    
    ArrayCentroids = np.asarray(ListCentroids, np.float32)
    
    #2. Filename_LSC:
    Reader_LSC = io.open(Filename_LSC, 'r', encoding='utf-8-sig', newline='\n', errors='ignore')
    ListSubCluster = list()
    for line in Reader_LSC:
        ListSubCluster.append(int(line))
    
    #3. Filename_LCLP:
    Reader_LCLP = io.open(Filename_LCLP, 'r', encoding='utf-8-sig', newline='\n', errors='ignore')
    ListClusterListPoints = list()
    for line in Reader_LCLP:
        tokens = line.rstrip().split(' ')
        ListPoints = list()
        for i in tokens:
            if (i != ''):
                ListPoints.append(int(i))
            else:
                break
        
        ListClusterListPoints.append(ListPoints)
    
    #4. Filename_Lac:
    Reader_Lac = io.open(Filename_Lac, 'r', encoding='utf-8-sig', newline='\n', errors='ignore')
    ListCentroids1 = list()
    for line in Reader_Lac: 
        tokens = line.rstrip().split(' ')
        X = np.array(tokens, dtype=np.float32)
        X1 = X.reshape((-1, EmbeddingSize))
        
        ListCentroids1.append(X1)
    
    ArrayCentroids1 = np.asarray(ListCentroids1, np.float32)
    ArrayArrayCentroids1 = ArrayCentroids1.reshape((-1, ListClusterSize, EmbeddingSize))
    
    ListArrayCentroids = ArrayArrayCentroids1.tolist()
    
    #5. Filename_Llclp:
    Reader_Llclp = io.open(Filename_Llclp, 'r', encoding='utf-8-sig', newline='\n', errors='ignore')
    ListListClusterListPoints = list()
    ListClusterListPoints1 = list()
    for line in Reader_Llclp: 
        tokens = line.rstrip().split(' ')
        X = np.array(tokens, dtype=np.int32)
        
        ListClusterListPoints1.append(X)
    
    for i in range(0, len(ListClusterListPoints1), ListClusterSize):
        ListListClusterListPoints.append(ListClusterListPoints1[i:int(i+ListClusterSize)])
        
    ModelVoronoi = Voronoi(ArrayCentroids, ListSubCluster, ListClusterListPoints, ListArrayCentroids, ListListClusterListPoints)
    ...
    return ModelVoronoi


'''
Метод быстрого поиска точки в сжатых ВП с использование ассиметричных кластеров Вороного.
'''
def VoronoiLookup(CurrentWE, ListCompressedWE, GlobalArrayCentroids, EmbeddingSize, ArrayCentroids, ListSubCluster, ListClusterListPoints, ListArrayCentroids1, ListListClusterListPoints1):
    #0. Создадим список точек, в которых мы будем добавлять индексы ВП:
    CurrentListPoints = list()
    #1. Берем 2 наиболее близких к CurrentWE центроида из ArrayCentroids:
    CurrentListID = Algebra.EuclidianMaxN(ArrayCentroids, CurrentWE, 2)
    if (ListSubCluster[CurrentListID[0]] == 0):
        CurrentListPoints.extend(ListClusterListPoints[CurrentListID[0]])
    else:
        ID = ListSubCluster[CurrentListID[0]] - len(ListCompressedWE)
        CurrentListID2a = Algebra.EuclidianMaxN(ListArrayCentroids1[ID], CurrentWE, 2)
        CurrentListPoints.extend(ListListClusterListPoints1[ID][CurrentListID2a[0]])
        CurrentListPoints.extend(ListListClusterListPoints1[ID][CurrentListID2a[1]])
    if (ListSubCluster[CurrentListID[1]] == 0):
        CurrentListPoints.extend(ListClusterListPoints[CurrentListID[0]])
    else:
        ID = ListSubCluster[CurrentListID[1]] - len(ListCompressedWE)
        CurrentListID2b = Algebra.EuclidianMaxN(ListArrayCentroids1[ID], CurrentWE, 2)
        CurrentListPoints.extend(ListListClusterListPoints1[ID][CurrentListID2b[0]])
        CurrentListPoints.extend(ListListClusterListPoints1[ID][CurrentListID2b[1]])
    
    #5. По списку индексов CurrentListPoints1 расжимаем точки-кандидаты:
    CurrentListCompressedWE = list()
    for i in CurrentListPoints:
        CurrentListCompressedWE.append(ListCompressedWE[i])
    CurrentArrayCompressedWE = np.asarray(CurrentListCompressedWE)
    CurrentArrayWE = Compressor.DecompressListWE(CurrentArrayCompressedWE, GlobalArrayCentroids, EmbeddingSize)
    #6. Ищем 1 точку-победитель:
    ID = Algebra.EuclidianMax(CurrentArrayWE, CurrentWE)
    CurrentID = CurrentListPoints[ID]
    
    ...
    return CurrentID


'''
Метод быстрого поиска СПИСКА точек в сжатых ВП с использование ассиметричных кластеров Вороного.
'''
def VoronoiLookupN(CurrentWE, ListCompressedWE, GlobalArrayCentroids, EmbeddingSize, ArrayCentroids, ListSubCluster, ListClusterListPoints, ListArrayCentroids1, ListListClusterListPoints1, N):
    #0. Создадим список точек, в которых мы будем добавлять индексы ВП:
    CurrentListPoints = list()
    #1. Берем 2 наиболее близких к CurrentWE центроида из ArrayCentroids:
    CurrentListID = Algebra.EuclidianMaxN(ArrayCentroids, CurrentWE, 2)
    if (ListSubCluster[CurrentListID[0]] == 0):
        CurrentListPoints.extend(ListClusterListPoints[CurrentListID[0]])
    else:
        ID = ListSubCluster[CurrentListID[0]] - len(ListCompressedWE)
        CurrentListID2a = Algebra.EuclidianMaxN(ListArrayCentroids1[ID], CurrentWE, 2)
        CurrentListPoints.extend(ListListClusterListPoints1[ID][CurrentListID2a[0]])
        CurrentListPoints.extend(ListListClusterListPoints1[ID][CurrentListID2a[1]])
    if (ListSubCluster[CurrentListID[1]] == 0):
        CurrentListPoints.extend(ListClusterListPoints[CurrentListID[0]])
    else:
        ID = ListSubCluster[CurrentListID[1]] - len(ListCompressedWE)
        CurrentListID2b = Algebra.EuclidianMaxN(ListArrayCentroids1[ID], CurrentWE, 2)
        CurrentListPoints.extend(ListListClusterListPoints1[ID][CurrentListID2b[0]])
        CurrentListPoints.extend(ListListClusterListPoints1[ID][CurrentListID2b[1]])
    
    #5. По списку индексов CurrentListPoints1 расжимаем точки-кандидаты:
    CurrentListCompressedWE = list()
    for i in CurrentListPoints:
        CurrentListCompressedWE.append(ListCompressedWE[i])
    CurrentArrayCompressedWE = np.asarray(CurrentListCompressedWE)
    CurrentArrayWE = Compressor.DecompressListWE(CurrentArrayCompressedWE, GlobalArrayCentroids, EmbeddingSize)
    #6. Ищем 1 точку-победитель:
    ListID = Algebra.EuclidianMaxN(CurrentArrayWE, CurrentWE, N)
    ListCurrentID = list()
    for ID in ListID:
        ListCurrentID.append(CurrentListPoints[ID])
    
    ...
    return ListCurrentID
