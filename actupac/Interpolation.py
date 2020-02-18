import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from .General import Initial

class Interpolation(Initial):  
    
    def __init__(self, mu=0, sigma=1):
        
        Initial.__init__(self, mu, sigma)
        
    def piecewise_linear_single(self,t1,t2,y1,y2,n):
        """Function to calcuulate piecewise linear interest yields between t1 and t2

        Args:
            t1: time to maturity t1
            t2: time to maturity at the next period t2
            y1: yields at t1
            y2: yields at the next period t2
            n: number of interpolated points
        Returns:
            original interest rate with interpolated rates

        """    
        interpo = []
        step = (t2-t1)/(n+1)
        pt = []
        years = []
        
        # Years
        for i in range(n):
            temp = t1+step*(i+1)
            years.append(temp)
        years.insert(0,t1)    
            
        # interpolate yields
        for i in range (n):
            temp = round((y1+step*(i+1)/(t2-t1)*(y2-y1)),5) # take 5 decimal numbers
            interpo.append(temp)
        interpo.insert(0,y1) 
        
        # interpolate discount rates
        for i in range (len(interpo)):
            temp = math.exp(-(t1+ step*(i))* interpo[i]/100)
            pt.append(temp)
        
        
        constant = pd.DataFrame([years,interpo,pt]).T
        constant.columns=('Years','Yields','Pt')
        return constant
    
    def piecewise_linear(self,n):
        """Function to calcuulate piecewise linear interest yields, discount rates and forward rates for all data.

        Args:
            n: number of interpolated points
        Returns:
            original interest rate with interpolated rates

        """ 
        all_df = []

        df = self.data
        for i in range (len(df)-1):
            t1 = df['Years'][i]
            t2 = df['Years'][i+1]
            y1 = df['Yields'][i]
            y2 = df['Yields'][i+1]
            all_df.append(self.piecewise_linear_single(t1,t2,y1,y2,n))

        
        all_df = pd.concat(all_df)
        
        # Deal with the last year
        year_end = df['Years'].iloc[-1]
        yield_end = df['Yields'].iloc[-1]
        Pt_end = math.exp(-year_end * yield_end/100)
        temp = list([year_end,yield_end,Pt_end])
        temp = pd.DataFrame(temp).T
        temp.columns = all_df.columns
        all_df = pd.concat([all_df, temp])
        all_df = all_df.reset_index(drop=True)
        
        ft = []   
        years = all_df['Years']
        
        # interpolate forward rate
        for i in range (len(all_df)-1):
            step = years[i+1]-years[i]
            temp = 1/step*(all_df['Pt'][i]/all_df['Pt'][i+1]-1)*100
            ft.append(temp)      
        ft.append(np.nan)
        all_df['Ft'] = ft
        
        self.linear = all_df
        return self.linear
    
    def piecewise_linear_plot(self):
        """Function to plot piecewise linear interest yields, discount rates and forward rates for all data.

        Args:
            None
        Returns:
            None

        """   
        plt.figure(figsize=(6,6.5))
        plt.subplot(2,1,1)
        plt.plot(self.linear['Years'],self.linear['Yields'],label='Yields')
        plt.plot(self.linear['Years'],self.linear['Ft'],label='Forward rates')
        plt.scatter(self.data['Years'], self.data['Yields'], s=10, c='red', marker='o',label = 'Known yields')
        plt.legend()
        plt.title('Intropolation with piecewise linear model')
        plt.xlabel('Years')
        plt.ylabel('Rates(%)') 
        
        plt.subplot(2,1,2)
        plt.plot(self.linear['Years'],self.linear['Pt'],label='Discount rates')
        plt.legend()
        plt.xlabel('Years')
        plt.ylabel('Rates(%)')
        plt.show()

    def piecewise_constant_single(self,t1,t2,y1,y2,n):
        """Function to calcuulate piecewise constant interest yields between t1 and t2

        Args:
            t1: time to maturity t1
            t2: time to maturity at the next period t2
            y1: yields at t1
            y2: yields at the next period t2
            n: number of interpolated points
        Returns:
            original interest rate with interpolated rates

        """    
        interpo = []
        step = (t2-t1)/(n+1)
        pt = []
        years = []
        
        # Years
        for i in range(n):
            temp = t1+step*(i+1)
            years.append(temp)
        years.insert(0,t1)    
            
        # interpolate yields
        for i in range (n):
            t = t1 + step * (i+1)
            temp = (y1*(t2/t-1)/(t2/t1-1)+y2*(1-t1/t)/(1-t1/t2))
            interpo.append(temp)
        interpo.insert(0,y1) 
        
        # interpolate discount rates
        for i in range (len(interpo)):
            temp = math.exp(-(t1+ step*(i))* interpo[i]/100)
            pt.append(temp)
        
        
        constant = pd.DataFrame([years,interpo,pt]).T
        constant.columns=('Years','Yields','Pt')
        return constant

    def piecewise_constant(self,n):
        """Function to calcuulate piecewise constant interest yields, discount rates and forward rates for all data.

        Args:
            n: number of interpolated points
        Returns:
            original interest rate with interpolated rates

        """ 
        all_df = []

        df = self.data
        for i in range (len(df)-1):
            t1 = df['Years'][i]
            t2 = df['Years'][i+1]
            y1 = df['Yields'][i]
            y2 = df['Yields'][i+1]
            all_df.append(self.piecewise_constant_single(t1,t2,y1,y2,n))

        
        all_df = pd.concat(all_df)
        
        # Deal with the last year
        year_end = df['Years'].iloc[-1]
        yield_end = df['Yields'].iloc[-1]
        Pt_end = math.exp(-year_end * yield_end/100)
        temp = list([year_end,yield_end,Pt_end])
        temp = pd.DataFrame(temp).T
        temp.columns = all_df.columns
        all_df = pd.concat([all_df, temp])
        all_df = all_df.reset_index(drop=True)
        
        ft = []   
        years = all_df['Years']
        
        # interpolate forward rate
        for i in range (len(all_df)-1):
            step = years[i+1]-years[i]
            temp = 1/step*(all_df['Pt'][i]/all_df['Pt'][i+1]-1)*100
            ft.append(temp)      
        ft.append(np.nan)
        all_df['Ft'] = ft
        
        self.constant = all_df
        return self.constant
    
    def piecewise_constant_plot(self):
        """Function to plot piecewise constant interest yields, discount rates and forward rates for all data.

        Args:
            None
        Returns:
            None

        """   
        plt.figure(figsize=(6,6.5))
        plt.subplot(2,1,1)
        plt.plot(self.constant['Years'],self.constant['Yields'],label='Yields')
        plt.plot(self.constant['Years'],self.constant['Ft'],label='Forward rates')
        plt.scatter(self.data['Years'], self.data['Yields'], s=10, c='red', marker='o',label = 'Known yields')
        plt.legend()
        plt.title('Intropolation with piecewise constant model')
        plt.xlabel('Years')
        plt.ylabel('Rates(%)') 
        
        plt.subplot(2,1,2)
        plt.plot(self.constant['Years'],self.constant['Pt'],label='Discount rates')
        plt.legend()
        plt.xlabel('Years')
        plt.ylabel('Rates(%)')
        plt.show()

    def cubic(self,n):
        """Function to calcuulate cubic spline interest yields

        Args:
            n: number of interpolated points
        Returns:
            original interest rate with interpolated rates

        """    
        interpo = []
        pt = []
        years_inter = []
        df = self.data
        years = df['Years']
        yields = df['Yields']
        
        # Years
        for j in range(len(years)-1):
            t1 = years[j]
            t2 = years[j+1]
            step = (t2-t1)/(n+1)
            for i in range(n):
                temp = t1+step*(i+1)
                years_inter.append(temp) 
            years_inter.insert(j*n+j,t1)
        years_inter.append(years.iloc[-1])
        
        # interpolate yields
        # calculate the second derivative with the matrix
        right = []
        left = np.zeros((1,30))
        left[0][0] = 1
        for i in range(len(self.data['Years'])-2):
            i=i+1
            step1 = self.data['Years'][i]-self.data['Years'][i-1]
            step2 = self.data['Years'][i+1]-self.data['Years'][i]

            # calculate left
            temp1 = np.zeros((1,30))
            temp1[0][i-1] = step1
            temp1[0][i] = 2*(step1+step2)
            temp1[0][i+1] = step2
            left = np.vstack((left,temp1))

            # calculate right
            temp2 = ((self.data['Yields'][i+1]-self.data['Yields'][i])/step2-(self.data['Yields'][i]-self.data['Yields'][i-1])/step1)*6
            right.append(temp2)

        right.insert(0,0)
        right.append(0)
        right = np.array(right)
        last = np.zeros((1,30))
        last[0][-1] = 1
        left = np.vstack((left,last))
        second_term = np.linalg.solve(left,right)
        
        # calculate the yields
        for i in range (len(years)-1):
            t1 = years[i]
            t2 = years[i+1]
            y1 = yields[i]
            y2 = yields[i+1]
            step = (t2-t1)/(n+1)
            for j in range(n+1):
                t = t1 + step * j
                temp = (t2-t)/(t2-t1)*(y1+1/6*second_term[i]*(t-t1)*(t-2*t2+t1))+(t-t1)/(t2-t1)*(y2-1/6*second_term[i+1]*(t2-t)*(t-2*t1+t2))
                interpo.append(temp)
        
        # interpolate discount rates
        for i in range (len(interpo)):
            temp = math.exp(-years_inter[i]* interpo[i]/100)
            pt.append(temp)
            
        # interpolate forward rate
        ft=[]
        for i in range (len(pt)-1):
            step = years_inter[i+1]-years_inter[i]
            temp = 1/step*(pt[i]/pt[i+1]-1)*100
            ft.append(temp)      
        ft.append(np.nan)
        
        cubic = pd.DataFrame([years_inter,interpo,pt,ft]).T
        cubic.columns=('Years','Yields','Pt','Ft')
        
        # Deal with the last year
        year_end = df['Years'].iloc[-1]
        yield_end = df['Yields'].iloc[-1]
        Pt_end = math.exp(-year_end * yield_end/100)
        step = cubic['Years'].iloc[-1] - cubic['Years'].iloc[-2]
        Ft_end = 1/step*(cubic['Pt'].iloc[-2]/Pt_end-1) * 100
        cubic['Yields'].iloc[-1] = yield_end
        cubic['Pt'].iloc[-1] = Pt_end
        cubic['Ft'].iloc[-2] = Ft_end
    
        self.cubic = cubic
        return self.cubic
            
    def cubic_plot(self):
        """Function to plot cubic spline interest yields, discount rates and forward rates for all data.

        Args:
            None
        Returns:
            None

        """   
        plt.figure(figsize=(6,6.5))
        plt.subplot(2,1,1)
        plt.plot(self.cubic['Years'],self.cubic['Yields'],label='Yields')
        plt.plot(self.cubic['Years'],self.cubic['Ft'],label='Forward rates')
        plt.scatter(self.data['Years'], self.data['Yields'], s=10, c='red', marker='o',label = 'Known yields')
        plt.legend()
        plt.title('Intropolation with cubic spline model')
        plt.xlabel('Years')
        plt.ylabel('Rates(%)') 
        
        plt.subplot(2,1,2)
        plt.plot(self.cubic['Years'],self.cubic['Pt'],label='Discount rates')
        plt.legend()
        plt.xlabel('Years')
        plt.ylabel('Rates(%)')
        plt.show()