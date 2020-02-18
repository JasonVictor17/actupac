import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

class Extrapolation():  
    
    def __init__(self,years,yields):
        """Initial the input years and yields
        
        Args: 
            years: a list of years
            yields: a list of yields
        Returns: 
            None
    
        """
        self.years = years
        self.yields = yields
        if len(self.years) != len(self.yields):
            print('Warning, the length of years and yields should be equal.')
    
    def dataframe(self):
        """transform a list to a dataframe
        
        Args: 
            None
            
        Returns: 
            a dataframe
    
        """
        
        dataframe = pd.DataFrame([self.years,self.yields]).T
        dataframe.columns = ('Years','Yields')
        self.dataframe = dataframe
        return self.dataframe
    
    def discount(self):
        """transform yields to discount rates
        
        Args: 
            None
            
        Returns: 
            discount rates
    
        """
        
        pt = []
        for i in range(len(self.yields)):
            temp = math.exp(-self.yields[i] * self.years[i]/100)
            pt.append(temp)
        self.discount = pt
        
        return(self.discount)
    
    def Smith_Wilson(self, n, max_year, UFR):
        """interpolate and extrapoate yields, discount rates and forward rates by the Smith Wilson curve.
        
        Args: 
            n: number of points that will be interpolate and extrapoate in each year.
            max_year: the maximum number of years that will be extrapoate
            UFR: the UFR rate with %
            
        Returns: 
            a dataframe with interpolated and extrapoated rates 
    
        """
        # Calculate the value of sw
        SW=[]
        years_array = np.array(self.years)
        for each in self.years:
            temp = n * np.array([np.tile(each,len(self.years)), years_array]).min(0)+0.5 * np.exp(-n*(each+years_array))-0.5 * np.exp(-n * np.abs(each - years_array))
            SW.append(temp)
        SW = np.array(SW)
        
        # Calculate the values of ITA
        right = self.discount() / np.exp(-UFR/100*years_array)-1
        ita = np.linalg.solve(SW,right)  
        
        # Inter and extro yields
        years_all = np.arange(0,max_year+n,n)
        SW_all = []
        for each in years_all:
            temp = n * np.array([np.tile(each,len(self.years)), years_array]).min(0)+0.5 * np.exp(-n*(each+years_array))-0.5 * np.exp(-n * np.abs(each - years_array))
            SW_all.append(temp)
        SW_all = np.array(SW_all)
        
        years_all = np.array(years_all)
        result_Pt=(1+(SW_all.dot(ita)))*np.array([np.exp(-UFR/100*years_all)])
        result_Yt = -1/years_all[1:]*np.log(result_Pt[0][1:len(years_all)])*100
        result_Yt = result_Yt.tolist()
        result_Yt.insert(0,np.nan)
        
        # Ft
        result_Pt_temp = result_Pt[0][1:len(years_all)]
        result_Ft = 1/n*(result_Pt[0][0:-1]/result_Pt_temp-1)*100
        result_Ft = result_Ft.tolist()
        result_Ft.append(np.nan)
        
        # dataframe
        years_all = years_all.tolist()
        result_Pt = result_Pt[0].tolist()
        df = {'Years':years_all,
             'Yields':result_Yt,
             'Discount rates':result_Pt,
             'Forward rates':result_Ft}
        self.Smith = pd.DataFrame(df)
        self.UFR = UFR
        return self.Smith
    
    def Smith_Wilson_plot(self): 
        """Function to plot Smith Wilson interest yields, discount rates and forward rates for all data.

        Args:
            None
        Returns:
            None

        """   
        plt.figure(figsize=(6,6.5))
        plt.subplot(2,1,1)
        plt.plot(self.Smith['Years'],self.Smith['Yields'],label='Yields')
        plt.plot(self.Smith['Years'],self.Smith['Forward rates'],label='Forward rates')
        plt.scatter(self.dataframe['Years'], self.dataframe['Yields'],c='red',s=10,label='Known yields')
        plt.axhline(y=self.UFR,ls=':',c='black')
        plt.text(self.Smith['Years'].iloc[-1]/2, self.UFR + 0.1, 'UFR = '+ str(self.UFR)+'%', fontsize=10)
        plt.legend()
        plt.title('Intropolation and extrapolation with Smith Wilson model')
        plt.xlabel('Years')
        plt.ylabel('Rates(%)') 
        
        plt.subplot(2,1,2)
        plt.plot(self.Smith['Years'],self.Smith['Discount rates'],label='Discount rates')
        plt.legend()
        plt.xlabel('Years')
        plt.ylabel('Rates(%)')
        plt.show()