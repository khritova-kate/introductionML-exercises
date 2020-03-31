from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale

def line_to_float (X, len_x, line_i):
    for i in range (len_x-1):
        X[line_i][i] = float(X[line_i][i])
    X[line_i][len_x - 1] = float(X[line_i][len_x - 1].split('\n')[0])
    return X

def CV_grade_kNN (X, y, kf):
    grade = 0
    grade_k = 0
    for k in range(1, 51):
        knn = KNeighborsClassifier( n_neighbors = k )
        M = cross_val_score(knn, X, y = y, cv = kf).mean()
        if (grade < M):
            grade = M
            grade_k = k
    return grade_k, grade


#считываем данные из файла
X = list()
y = list()
line_i = 0

with open('wine.data', 'r') as f:
    for line in f:
        y += [ int (line.split(',')[0])  ]
        X += [ line.split(',')[1:] ]
        len_x = len(X[0])
        X = line_to_float(X, len_x, line_i)
        line_i += 1
        
#print(y)
#print(X, "\n", len_x)

#оценка качества методом кросс-валидации по 5 блокам (с перемешиванием объектов)
kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

#kNN при k = 1, ... , 50 + оценка кросс-валидации
grade_k, grade = CV_grade_kNN(X, y, kf)
    
print('1. k of optimal grade ', grade_k)
print('2. Optimal grade      ', round( grade, 2 ))


#масштабирование признаков
X = scale(X)
grade_k, grade = CV_grade_kNN(X, y, kf)
    
print('3. k of optimal grade ', grade_k)
print('4. Optimal grade      ', round( grade, 2 ))

input()
    
    