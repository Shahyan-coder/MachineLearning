#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
class LogisticRegression():
    def __init__(self, lr, num_iters):
        self.lr = lr
        self.num_iters = num_iters
        self.acc_iters = None
        self.cost = None
        self.w = None
        self.b = None
        
    def sigmoid(self,linear_prediction):
        return 1 / (1+np.exp(-linear_prediction))
        
    def fit(self,X,y,thres):
        m,n_feat = X.shape
        self.w = np.zeros((n_feat,1))
        self.b = 0
        self.cost = 0
        self.acc_iters = 0
        for iteration in range(self.num_iters):
            linear_prediction = np.dot(X,self.w)+self.b
            logistic_prediction = self.sigmoid(linear_prediction)
            wj = (1/m)*np.dot(X.T,(logistic_prediction-y))
            grad_b = (1/m)*np.sum(logistic_prediction-y)
            
            self.w = self.w - self.lr*wj
            self.b = self.b - self.lr*grad_b
            prev_cost = self.cost
            self.cost = self.cost + (-1/m)*np.sum(y*np.log(logistic_prediction) + (1-y)*np.log(1-logistic_prediction))
            self.acc_iters = self.acc_iters + 1
            if self.cost - prev_cost < thres:
                break
                
    def iterations(self):
        return self.acc_iters
            
            
    def predict(self, X):
        linear_prediction = np.dot(X,self.w)+self.b
        logistic_prediction = self.sigmoid(linear_prediction)
        final_prediction = []
        for y in logistic_prediction:
            if y<=0.5:
                final_prediction.append(0)
            else:
                final_prediction.append(1)
        return final_prediction
            
            
        

