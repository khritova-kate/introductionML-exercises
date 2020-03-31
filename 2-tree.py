import pandas as pd
import numpy as np

#Решающие деревья для классификации и регрессии
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv('titanic.csv', index_col = 'PassengerId')
pd.DataFrame(data)

#удаление столбцов 
data.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'],
          axis = 'columns', inplace=True)

#ищем объекты, у к-рых тсутсьвуют признаки и удаляем их
max_range = len(data)    
j = 0
for i in range(max_range):
    if( np.isnan (data[  'Age' ].tolist()[j]) or
        np.isnan (data['Pclass'].tolist()[j]) or
        np.isnan (data[ 'Fare' ].tolist()[j]) or
        not( data['Sex'].tolist()[j])     ):
        
        data.drop([i+1], inplace=True)
    else:
        j += 1

#изменим згачения в 'Sex' на числа
data.loc[data['Sex'] == 'male', 'Sex'] = 1
data.loc[data['Sex'] == 'female', 'Sex'] = 0

print(data)

#берем целевую переменную и признаки
target   = data['Survived'].values
#print(target)
features = data[ ['Age', 'Fare', 'Pclass', 'Sex'] ].values
#print(features)

clf = DecisionTreeClassifier(random_state = 241)
clf.fit(features, target)

#массив 'важностей'
importances = clf.feature_importances_

print(list(data))
print(importances)

input()