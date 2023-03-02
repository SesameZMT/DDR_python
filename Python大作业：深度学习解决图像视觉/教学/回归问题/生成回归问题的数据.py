import random
import pandas as pd
import csv
import numpy as np

w = 0.235
b = 0.6785

l = np.zeros((1000,2))

list=[]

for i in range(0,999):
    x = random.uniform(0,100)
    y = w * x + b
    l[i][0] = x
    l[i][1] = y

test = pd.DataFrame(data=l)
test.to_csv('data3.csv',index=False,header=False)