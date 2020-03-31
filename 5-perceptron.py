import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv('perceptron-train.csv', header = None)
df_test  = pd.read_csv('perceptron-test.csv' , header = None)

#print(df_train)
#print(df_test)

X_train = df_train.drop([0],   axis = 'columns').values
y_train = df_train[0].values

X_test  = df_test.drop([0],   axis = 'columns').values
y_test  = df_test[0].values

#обучаем персептрон
clf = Perceptron(random_state = 241, max_iter=5, tol=None)
clf.fit(X_train, y_train)

#предсказание и доля правильных ответов
predictions = clf.predict(X_test)
acc_1 = accuracy_score(y_test, predictions)

#масштабируем данные
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

clf.fit(X_train_scaled, y_train)
predictions = clf.predict(X_test_scaled)
acc_2 = accuracy_score(y_test, predictions)

print(acc_2,'-',acc_1, '=', round(acc_2 - acc_1,3))

input()