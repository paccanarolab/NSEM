import numpy as np
from scipy import sparse
from collections import defaultdict
from itertools import zip_longest as izip
import warnings
from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix,
    csr_matrix, isspmatrix, dok_matrix, lil_matrix, bsr_matrix)
warnings.simplefilter('ignore',SparseEfficiencyWarning)

class Nsem:
    def __init__(self, Ytrain, Ytest, tolX = 1e-2, var = 0.01, max_iters = 1000, gamma = 10000):
        self.tolX = tolX
        self.variance = var
        self.max_iters = max_iters
        self.gamma = gamma
        self.Y = Ytrain
        self.Ytest = Ytest
        # Get the dimensions
        (self.nUsers,self.nItems) = self.Y.shape

    def train(self, L2beta, L1lambda):
        
        # initialization   
        W = np.random.uniform(0,np.sqrt(self.variance),(self.nItems,self.nItems))
       
        W0 = W;
        # numerator
        numer = np.dot(self.Y.transpose(), self.Y)
        numer = numer.toarray()    
        
        # Identity
        I = np.identity(self.nItems);
           
        # get machine precision eps
        epsilon = np.finfo(float).eps  
        sqrteps = np.sqrt(epsilon);
        
        for iter in range(self.max_iters):
            denom = np.dot((numer + L2beta), W) + L1lambda + epsilon + self.gamma*I
           
            W = np.multiply(W, np.divide(numer, denom))
            
            # Delete negative values due to machine precision.
            W.clip(min = 0)
            
            # Get the max change in W      
            dw = np.amax(np.abs(W-W0))/(sqrteps + np.amax(np.abs(W0)));
            
            if dw <= self.tolX:
                # print('Iter', iter, 'dw', dw)
                break
            
            W0 = W;
        # values lower that this will be considered as zeros
        W[W < 1e-5] = 0
        WCSR = sparse.csr_matrix(W)
        return WCSR

    def zero(self,YhatNoZeros):
        YhatZeroed = YhatNoZeros
        for user, row in enumerate(self.Y):
            for itemIndex in row.indices:
                YhatZeroed[user,itemIndex] = 0
        return YhatZeroed

    # Sorts the movies per user according to their reccomendation score. 
    # If you have the need to calculate a top bigger than 25 just change that number.
    def sort_yhat(self,m):
        m = m.tocoo()
        sorted_rows = defaultdict(list)
        tuples = izip(m.row, m.col, m.data)
        for i in sorted(tuples, key=lambda x: (x[0], x[2])):
            sorted_rows[i[0]].append((i[1], i[2]))
        clippedSorted_rows = defaultdict()
        for user, items in sorted_rows.items():
            clippedSorted_rows[user] = items[-25:]
        return clippedSorted_rows
         
    def hit_rate(self, W, topN):
        Yhat = self.zero(self.Y.dot(W))
        sorted_rows = self.sort_yhat(Yhat)
        hitCount = 0
        for user in range(len(sorted_rows)):
            if self.Ytest[user].indices[0] in [i[0] for i in sorted_rows[user][-topN:]]:
                hitCount += 1
        return hitCount/self.nUsers      

    def avg_rec_hit_rate(self, W, topN):
        Yhat = self.zero(self.Y.dot(W))
        sorted_rows = self.sort_yhat(Yhat)
        hitCount = 0
        for user in range(len(sorted_rows)):
            topMovies = [i[0] for i in sorted_rows[user][-topN:]]
            if self.Ytest[user].indices[0] in topMovies:
                hitCount += (1/(topN - topMovies.index(self.Ytest[user].indices[0])))
        return hitCount/self.nUsers 