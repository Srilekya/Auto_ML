import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report

class Classifier:
    def __init__(self, train_x = None, train_y = None, test_x = None, test_y = None):
        self.xtrain = train_x
        self.ytrain = train_y
        self.xtest = test_x
        self.ytest = test_y
        self.models = [LogisticRegression(max_iter = 5000), DecisionTreeClassifier(), RandomForestClassifier(), GaussianNB(), KNeighborsClassifier(), AdaBoostClassifier()]
        self.names = ['Logistic', 'DT', 'RF', 'GNB', 'KNN', 'ADB']

    def train(self):
        train_results = pd.DataFrame(columns = ['Models', 'Train_accuracy'])
        for name, model in zip(self.names, self.models):
            print((f'Training {name} model'))
            model.fit(self.xtrain, self.ytrain)
            train_acc = model.score(self.xtrain,self.ytrain)
            train_results = train_results.append({'Models': name, 'Train_accuracy': train_acc}, ignore_index= True)
            return train_results

    def confusion_matrix_train(self):
        cons = pd.DataFrame(columns = ['Models', 'Confusion_Matrix'])
        for name, model in zip (self.names, self.models):
            print((f'Training {name} model '))
            model.fit(self.xtrain, self.ytrain)
            y_pred = model.predict(self.xtrain)
            con = confusion_matrix(y_pred, self.ytrain)
            cons = cons.append({'Models': name, 'Confusion_Matrix': con}, ignore_index = True)
            return cons

    def confusion_matrix_test(self):
        cons = pd.DataFrame(columns = ['Models', 'Confusion_Matrix'])
        for name, model in zip (self.names, self.models):
            print((f'Testing {name} model'))
            model.fit(self.xtest, self.ytest)
            y_pred = model.predict(self.xtest)
            con = confusion_matrix(y_pred, self.ytest)
            cons = cons.append({'Models': name, 'Confusion_Matrix': con}, ignore_index = True)
            return cons
