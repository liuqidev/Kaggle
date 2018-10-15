import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


obj=pd.Series(['c','a','d','a','a','b','b','c','c','c'])
# print(obj.unique())
count={}
for i in obj:
    type(i)
    print(i)
    # count[i]+=1

for e in count.items():
    print(e)