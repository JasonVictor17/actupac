import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

class Initial:
    
    def __init__(self, mu = 0, sigma = 1):
        
        self.mean = mu
        self.stdev = sigma
        self.data = []

    def calculate_mean(self):
    
        """Method to calculate the mean of the data set.
        
        Args: 
            None
        
        Returns: 
            float: mean of the data set
    
        """
        
        self.mean = round(sum(self.data['Yields'])/len(self.data['Yields']),4)
        
        return self.mean
                
    def calculate_stdev(self, sample = False):

        """Method to calculate the standard deviation of the data set.
        
        Args: 
            sample (bool): whether the data represents a sample or population
        
        Returns: 
            float: standard deviation of the data set
    
        """
        
        if sample:
            n = len(self.data['Yields'])-1
        else:
            n = len(self.data['Yields'])
        
        mean = self.mean
        sigma = 0
        for each in self.data['Yields']:
            sigma += (each-mean) ** 2
        
        sigma = math.sqrt(sigma/n)
  
        self.stdev = round(sigma,3)
        return self.stdev
    
    def read_data(self, file_name):
    
        """Function to read in data from a csv file. The csv file should have
        two numbers per line, the first one is time to maturity, the second one is yields. 
        The numbers are stored in the data attribute.
                
        Args:
            file_name (string): name of a file to read from
        
        Returns:
            None
        
        """
            
        data_list = pd.read_csv(file_name,header = None)
        data_list.columns =  ('Years', 'Yields')
        self.data = data_list
        self.mean = self.calculate_mean()
        self.stdev = self.calculate_stdev()
        
        return self.data
    
    def plot_histogram(self):
        """Method to output a histogram of the instance variable data using 
        matplotlib pyplot library.
        
        Args:
            None
            
        Returns:
            None
        """
        
        plt.hist(self.data['Yields'],alpha=0.5,rwidth=0.8,edgecolor='black')
        plt.title('Distribution of the interest rates')
        plt.xlabel('interest rates')
        plt.ylabel('conuts') 
    
    def discount_rate(self):
        
        """Method to output discount rates based on given yields.
         
        Args:
            None
            
        Returns:
            discount rates
        """
        pt = []
        for i in range(len(self.data)):
            temp = math.exp(-self.data['Years'][i] * self.data['Yields'][i]/100)
            pt.append(temp)
        self.discount_rate = pt
        
        return(self.discount_rate)