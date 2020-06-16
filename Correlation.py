import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import spearmanr
from collections import Counter
import math
from scipy.stats import chi2_contingency
import numpy as np

data = pd.read_csv('C:/Users/User/Documents/Python/DS/Data Cleaning/modified_health.csv')
print(data.head())
print('Finding predictive variables')
print('================================================================================')
class ChiSquare:
    def __init__(self, dataframe):
        self.data = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfObserved = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        print("The p value is ")
        print(self.p)
        if self.p<alpha:
            result="{0} is IMPORTANT for Prediction".format(colX)
        else:
            result="{0} is NOT an important predictor.".format(colX)
        print(result)
        
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.data[colX].astype(str)
        Y = self.data[colY].astype(str)
        
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        self._print_chisquare_result(colX,alpha)

data['dummyCat'] = np.random.choice([0, 1], size=(len(data),), p=[0.5, 0.5])
cT = ChiSquare(data)
testColumns = ['Age','Gender','High Blood Pressure (Yes/No)','Tobacco Use (Yes/No)','Fruits & Vegetable Consumption',
               'Sugar-Sweetened Beverage Consumption','Exercise']
for var in testColumns:
    cT.TestIndependence(colX=var,colY="Heart Disease (Yes/No)" )

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
print('Finding association')
print('================================================================================')
testColumns2 = [data['Gender'],data['High Blood Pressure (Yes/No)'],data['Tobacco Use (Yes/No)'],
               data['Fruits & Vegetable Consumption'],data['Sugar-Sweetened Beverage Consumption'],data['Exercise']]    
for var2 in testColumns2:
    print(cramers_v(var2,data["Age"]))
print('Finding association')
print('================================================================================')
testColumns2 = [data['Age'],data['High Blood Pressure (Yes/No)'],data['Tobacco Use (Yes/No)'],
               data['Fruits & Vegetable Consumption'],data['Sugar-Sweetened Beverage Consumption'],data['Exercise']]    
for var2 in testColumns2:
    print(cramers_v(var2,data["Gender"]))
print('Finding association')
print('================================================================================')
testColumns2 = [data['Age'],data['Gender'],data['Tobacco Use (Yes/No)'],
               data['Fruits & Vegetable Consumption'],data['Sugar-Sweetened Beverage Consumption'],data['Exercise']]    
for var2 in testColumns2:
    print(cramers_v(var2,data["High Blood Pressure (Yes/No)"]))
print('Finding association')
print('================================================================================')
testColumns2 = [data['Age'],data['Gender'],data['High Blood Pressure (Yes/No)'],
               data['Fruits & Vegetable Consumption'],data['Sugar-Sweetened Beverage Consumption'],data['Exercise']]    
for var2 in testColumns2:
    print(cramers_v(var2,data["Tobacco Use (Yes/No)"]))
print('Finding association')
print('================================================================================')
testColumns2 = [data['Age'],data['Gender'],data['High Blood Pressure (Yes/No)'],
               data['Tobacco Use (Yes/No)'],data['Sugar-Sweetened Beverage Consumption'],data['Exercise']]    
for var2 in testColumns2:
    print(cramers_v(var2,data["Fruits & Vegetable Consumption"]))
print('Finding association')
print('================================================================================')
testColumns2 = [data['Age'],data['Gender'],data['High Blood Pressure (Yes/No)'],
               data['Tobacco Use (Yes/No)'],data['Fruits & Vegetable Consumption'],
                data['Exercise']]    
for var2 in testColumns2:
    print(cramers_v(var2,data["Sugar-Sweetened Beverage Consumption"]))
print('Finding association')
print('================================================================================')
testColumns2 = [data['Age'],data['Gender'],data['High Blood Pressure (Yes/No)'],
               data['Tobacco Use (Yes/No)'],data['Fruits & Vegetable Consumption'],
                data['Sugar-Sweetened Beverage Consumption']]    
for var2 in testColumns2:
    print(cramers_v(var2,data["Exercise"]))
