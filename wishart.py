#!/usr/bin/env python
# coding: utf-8

# In[6]:


import math
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from operator import mul


# In[185]:


class wishart:
    """ Calculate the mean, Covariance  and correlation matrix and density  of a Wishart
    distribution. It also calculates the multivariate gamma function.
    Todo: Simulation of a Wishart distribution. Redo  everything for a Riesz distribution and 
    graphical Wishart distributions.
    
    Attributes:
        dimension (int)
        mean (matrix)
        covariance ( matrix)
        correlation ( matrix)
        characteristic (function)
        multivariate_gamma (function)
        pdf (function)
    """
    def __init__(self, dimension, Lambda=None, Sigma= None):
        """
        Args:
            dimension (int):
            Lambda (float): should be greater than (dimension-1)/2.
            Sigma (positive definite matrix)            
        """
        self.dimension = dimension
        self.Lambda = Lambda
        self.Sigma = Sigma
        if Sigma is None:
            self.Sigma = np.identity(dimension,dtype=float) #Set default value of Sigma to identity matrix
        self.K = np.linalg.inv(self.Sigma) # Inverse of Sigma
        self.data = [] #positive definite matrices
        self.mean = self.calculate_mean() 
        self.covariance = self.calculate_covariance()
        self.correlation = self.calculate_correlations()
        self.characteristic = self.calculate_characteristic
        self.multivariate_gamma = self.calculate_gamma
        self.pdf = self.calculate_density
        
    def calculate_mean(self):
        """Calculates the mean of the distribution from the given parameters
        Args: None
        returns: 
            mean (positive definite matrix)
        
        """
        return self.Lambda*self.K
    
    def calculate_covariance(self):
        
        return self.Lambda * np.kron(self.K, self.K)
    
    def calculate_correlations(self):
        covariance = self.covariance
        denom = np.sqrt(np.diagonal(covariance))
        corr = covariance/denom
        for raw in range(self.dimension):
            corr[raw] = corr[raw]/denom[raw]
        return corr
            
    def calculate_characteristic(self):
        """Calculates the characteristic function of the cone of positive definite matrices.
        Args:
            X (positive definite matrix)
        Returns:
            (float)
        """
        X = self.Sigma
        return (np.linalg.det(X))**((self.dimension+1)/2)
    
    def calculate_gamma(self):
        """Calculates the gamma function on the cone of positive definite matrices.
        Args: 
            None:
        Returns:
            (float)
        """
        s = self.Lambda
        gamma_list = [math.gamma(s-(i-1)/2) for i in range(1,self.dimension+1)]
        return (math.pi)**(self.dimension*(self.dimension-1)/4)*reduce(mul, gamma_list)
    
    def calculate_density(self, X):
        """Calculates the density function of the Wishart distribution.
        Args:
            X (positive definite matrix)
        Returns:
            pdf (float)
            
        """
        s = self.Lambda
        XSigma =  np.matmul(X,self.Sigma)
        unnormalized_pdf = np.exp(-np.trace(XSigma))* np.linalg.det(XSigma)**s*self.calculate_characteristic()
        pdf =  unnormalized_pdf/self.multivariate_gamma()  
        return pdf 

