import numpy as np
from sklearn.cluster import KMeans

class word (object):
  def __init__(self, ListWE, Centroid, ListTokenID, Text):
    self.ListWE = ListWE
    self.Centroid = Centroid
    self.ListTokenID = ListTokenID
    self.Text = Text
    
  def GetCentroid(self, ArrayWE):
    if (len(self.ListTokenID) == 1):
      self.Centroid = np.array(ArrayWE[self.ListTokenID[0]])
    else:
      CurrentArrayWE = []
      for j in range(len(self.ListTokenID)):
        CurrentArrayWE.append(ArrayWE[self.ListTokenID[j]])
      kmeans = KMeans(n_clusters=1, random_state=0).fit(CurrentArrayWE)
      CentroidWE = kmeans.cluster_centers_
      self.Centroid = np.array(CentroidWE)
    
