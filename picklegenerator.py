import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

# Load in data from the csv file
data = pd.read_csv('data/Cleveland.data.csv')
# We just removes all the instances which have missing values.
data = data[data['ca'] != '?']
data = data[data['thal'] != '?']
# 上面的操作会导致ca、thal这两列转换为字符串 而不是数字，从而导致后面的错误
# 这里的作用是将其转换为数字
data['ca'] = data['ca'].astype(int)
data['thal'] = data['thal'].astype(int)

# Convert feature values from strings to integers.
from sklearn.preprocessing import LabelEncoder
labelencoders = []
for col in data.columns:
    # labelencoder每次的fit操作，会清空之前的结果，所以所有的列都要记录
    labelencoder=LabelEncoder()
    data[col] = labelencoder.fit_transform(data[col])
    labelencoders.append(labelencoder)

# Set up the X and y variables
# 取前13列数据
X = data.iloc[:,0:13]
y = data.iloc[:,13]

LogReg = LogisticRegression()

# Train the model using the training set
LogReg.fit(X, y)
yy = LogReg.predict(X)


# Save the trained decision tree into a pickle file
with open('pickledata/logres.pickle','wb') as f:
    # labelencoder同样需要保存，所有列的labelencoder
    pickle.dump((LogReg,labelencoders),f)

# pickle_in = open('pickledata/logres.pickle','rb')
# clf = pickle.load(pickle_in)