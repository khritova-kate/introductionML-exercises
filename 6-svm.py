import pandas as pd
from sklearn.svm import SVC

df = pd.read_csv('svm-data.csv', header = None)
print (df)

X = df.drop([0], axis = 1).values
y = df[0].values

svc = SVC(C = 100000, random_state=241, kernel='linear')
print( svc.fit(X, y).support_)
