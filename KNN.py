#!/usr/bin/env python
# coding: utf-8

# In[4]:


from collections import Counter
import numpy as np
class KNN:
    def __init__(self,k):
        self.k = k
        
    def fit(self,XTrain,yTrain):
        self.XTrain = XTrain
        self.yTrain = yTrain
    
    def _distance(self,x1,x2):
        d = np.sqrt((np.sum(x1-x2)**2))
        return d
        
    def predict(self,Xtest):
        labelPredictions = []
        for x in Xtest:
            labelPredictions.append(self._predict(x))
        return labelPredictions
    
    def _predict(self,x):
        allDistances = []
        for x2 in self.XTrain:
            dist = self._distance(x,x2)
            allDistances.append(dist)
            
        closest_k_indices = np.argsort(allDistances)[:self.k]
        closest_y_values = []
        for i in closest_k_indices:
            closest_y_values.append(int(self.yTrain[i]))
            
        #print(closest_y_values)
      
        majority = Counter(closest_y_values).most_common()
        return majority[0][0]
        
        
        
    


# In[ ]:




