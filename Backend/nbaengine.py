
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix

college_data = pd.read_csv('college_player-2020-21.csv')
draft_data = pd.read_csv('draft-2020-2021.csv')

#dummies
conference = pd.get_dummies(college_data['Conference'])
player_class = pd.get_dummies(college_data['Class'])
position = pd.get_dummies(college_data['Pos'])
school = pd.get_dummies(college_data['School'])

college_data = college_data.drop('Conference', axis=1)
college_data = college_data.drop('Class', axis=1)
college_data = college_data.drop('Pos', axis=1)
college_data = college_data.drop('School', axis=1)

college_data = pd.concat([college_data, conference], axis=1)
college_data = pd.concat([college_data, player_class], axis=1)
college_data = pd.concat([college_data, position], axis=1)
college_data = pd.concat([college_data, school], axis=1)


x = 0

college_data.drop(x, axis='index', inplace=True)
print(college_data)
while x<len(college_data):
    row = list(college_data.iloc[x])
    null = pd.isna(row)
    if True in null:
        print(x)
        college_data.drop(x, axis='index')
        print(college_data)
    x += 1

drafted = list()
for x in range(len(college_data)):
    row = college_data.iloc[x]
    if draft_data['Player'].str.contains(row['Player']).any():
        drafted.append(1)
    else:
        drafted.append(0)
college_data['Drafted'] = drafted

college_data = college_data.sample(frac=1).reset_index(drop=True)
college_data = college_data.set_index('Player')

X = college_data.drop('Drafted', axis=1)
Y = college_data['Drafted']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

mlp = MLPClassifier(hidden_layer_sizes=(3,2), activation='relu')
mlp.fit(X_train,Y_train)

pred = mlp.predict(X_test)
