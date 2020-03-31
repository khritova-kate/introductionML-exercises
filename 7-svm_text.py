import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score

newsgroups = datasets.fetch_20newsgroups(
                    subset = 'all', 
                    categories = ['alt.atheism', 'sci.space']
             )

""" Вычисляем TF-IDF признаки ВСЕХ текстов (cтатистическая мера, используемая
    для оценки важности слова в контексте документа, являющегося частью
    коллекции документов) """
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target

#print(X)
#print(y)

"""
# область "поиска" параметра C
grid = {'C': np.power(10.0, np.arange(-5, 6))}

cv = KFold(n_splits = 5, shuffle = True, random_state = 241)
clf = SVC(kernel = 'linear', random_state = 241)

# поиск оптимального C для SVC с помощью кросс-валидации по 5 блокам
gs = GridSearchCV(clf, grid, scoring = 'accuracy', cv = cv)
gs.fit(X, y)
C_opt = gs.best_params_['C']
###print(best_C)
"""
C_opt = 1

# Обучаем SVM по всей выборке с оптимальным C
clf = SVC(C = C_opt, random_state = 241, kernel = 'linear')
clf.fit(X,y)

# 10 слов с наибольшим абсолютным значением веса
feature_mapping = vectorizer.get_feature_names()
feature = clf.coef_

feature = np.absolute(feature.toarray())
"""
print(sorted(feature[0], reverse = True)[:10])

[2.6631647884797105, 1.9203794002294927, 1.2546899512384038, 
 1.2491800073760078, 1.201611181752071,  1.1801315951388636, 
 1.13908083789883,   1.1306123446649008, 1.0970936466401477, 1.0293069271856938]
"""
words = list()
i = 0
while (len(words) < 10):
    if feature[0][i] > 1.0293:
        words.append(feature_mapping[i])
    i += 1

words = sorted(words)
for i in range(10):
    print(words[i], end = " ")
    



