import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import os

def ensemble(dir_name):
    path = dir_name
    df = []
    files= os.listdir(path)
    for file in files:
        df.append(pd.read_csv(path+'/'+file)['pred'])
    final_result=[]
    for i in range(len(df[0])):
        a,b,c=0,0,0
        for ans in df:
            if ans[i]==0:
                a+=1
            elif ans[i]==1:
                b+=1
            else:
                c+=1
        if a>=b and a>=c:
            final_result.append(0)
        elif b>=c:
            final_result.append(1)
        else:
            final_result.append(2)
    res_id =[]
    for i in range(1108):
        res_id.append('test_'+str(i))
    df_res = {
        'id': res_id,
        'pred':final_result
    }
    df_res=DataFrame(df_res)
    df_res
    df_res.to_csv('ensemble.csv',index=False,sep=',')

if __name__ == "__main__":
    ensemble('result')