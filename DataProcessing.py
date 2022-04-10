import pandas as pd
import datetime as dt
from dateutil import parser
from dateutil.relativedelta import relativedelta
import numpy as np
from numpy import timedelta64

# Read Data
data = pd.read_csv("econ_data_1.csv")
monthly = pd.read_csv("monthlydat.csv")


keys = ['INVESTMENT', 'GOVERNMENT']

with open('newfile.txt','w') as f:
    for key in keys:
        dates = []
        newdat = []
        for k in range(len(monthly['Date'])):
            dates.append(monthly['Date'].values[k])
            j = 0
            while j<3:
                f.write(str(monthly[key].values[k])+"\n")
                j +=1
        print(len(newdat))
        print(newdat)
        f.write("\n")
    
    

print(data)
print(monthly)


