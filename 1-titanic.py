import pandas
from collections import Counter

data = pandas.read_csv('titanic.csv', index_col = 'PassengerId')
pandas.DataFrame(data)

# Количство мужчин и женщин
print ( '1. ' , data['Sex'].value_counts(), '\n')

# Процентное отношение выживших
print ( '2. ' , round (100 * data['Survived'].value_counts().loc[1] / len(data) , 2), '\n')

# Доля пассажиров первого класса среди всех пассажиров
print ( '3. ' , round (100 * data['Pclass'].value_counts().loc[1] / len(data) , 2), '\n')

# Среднее и медиана возраста
print('4. ', round( data['Age'].mean(), 2) )
print('   ', round( data['Age'].median(), 2) , '\n')

# Корелляция Пирсона между признаками SibSp и Parch
print ( '5. ', round( data['SibSp'].corr(data['Parch']), 2), '\n')

# Самое популярное женское имя (всего 314 женщин)
name = data.sort_values(by = ['Sex'])['Name'].tolist()
name[:10]
name = name[0:314]
for i in range(len(name)):
    a = name[i].find('Miss.')
    if (a != -1):
        name[i] = name[i][a:].split()[1]
    else:
        b = name[i].find('(')
        if (b != -1):
            name[i] = name[i][b+1:].split()[0]
            name[i] = name[i].replace(')',"")
        else:
            name[i] = ""
            
print ( '6. ', Counter(name).most_common(1)[0], '\n')
