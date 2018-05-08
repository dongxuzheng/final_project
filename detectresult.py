import pickle

from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
import numpy as np



def predict(df:DataFrame):
    pickle_in = open('pickledata/logres.pickle','rb')
    lr,labelencoders = pickle.load(pickle_in)
    for i,col in enumerate(df.columns):
        # 之前的strlabel的问题会导致这里的错误
        v = df[col]
        labelencoder = labelencoders[i]
        try:
            df[col] = labelencoder.transform(v)
        except:
            df[col] = [-1]

    pickle_in.close()
    r = lr.predict(df)
    return r.ravel()[0]

# print(predict(DataFrame([[63,1,1,145,233,1,2,150,0,2.3,3,0]]))) # 0
# print(predict(DataFrame([[67,1,4,120,229,0,2,129,1,2.6,2,2]]))) # 1
# print(predict(DataFrame([[37,1,3,130,250,0,0,187,0,3.5,3,0]]))) # 0


# print(predict(DataFrame([[41,0,2,130,204,0,2,172,0,1.4,1,0]]))) # 0