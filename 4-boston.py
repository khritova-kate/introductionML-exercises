from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score
import numpy as np


in_data = load_boston()
#print(in_data['data'])
#print(in_data['target'])

data = scale(in_data['data'])
kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

grade = -100
grade_p = 0
for p in np.linspace(1, 10, 200):
    #добавляем в алгоритм веса, зависящие от расстояния
    #p - параметр в метрике Минковского
    knn = KNeighborsRegressor( n_neighbors = 5, metric = 'minkowski', p = p, 
                               weights = 'distance')
    #используем среднеквадратичную ошибку  качестве метрики качества
    M = cross_val_score(knn, data, y = in_data['target'], cv = kf, 
                        scoring = 'neg_mean_squared_error' ).mean()
    print(M)
    if (grade < M):
        grade = M
        grade_p = p
        
print('optimal p ', round(grade_p, 1))

input()