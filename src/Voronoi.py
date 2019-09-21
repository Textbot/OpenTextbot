
'''
Кластеризация пространства ВП для быстрого поиска по сжатым ВП средствами ассиметричных ячеек Вороного.
'''

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

