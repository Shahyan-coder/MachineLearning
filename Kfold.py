#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from statistics import mean
class Kfold:
    def kFoldTest(X,y,model,thres):
        size = len(y) // 5
        accs =[]
        for i in range(5):
            left = i*size
            right = (i+1)*size
            X_test = X[left:right]
            y_test = y[left:right]
            X_train = np.concatenate((X[:left],X[right:]))
            y_train = np.concatenate((y[:left],y[right:]))
            
            if thres == None:
                model.fit(X_train,y_train)
            else:
                model.fit(X_train,y_train,thres)
                iterations = model.iterations()
            y_pred = model.predict(X_test)
            j=0
            for i in range(len(y_pred)):
                if y_pred[i] == y_test[i]:
                    j = j+1
            accs.append(j/len(y_pred))
        acc = mean(accs)
        if thres == None:
            return acc
        else:
            return acc, iterations

            
            
        
        


# In[ ]:




