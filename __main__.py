import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.externals._packaging.version import parse
from classification.clf import *
from sklearn.model_selection import train_test_split

#from classification.clf import confusion_matrix_test, confusion_matrix_train, train

parser = argparse.ArgumentParser()
parser.add_argument('-d','--dataset', help = 'Dataset', type = str, default = None, required = True)
parser.add_argument('-l','--label', type = str, default = None, required = True)

args = parser.parse_args()
print('Reading the Dataset')

df = pd.read_csv(args.dataset)
print('Spliting the dataset into DV and IDV')

x = df.drop(labels = args.label, axis = 1)
y = df[args.label]

print('Spliting the Dataset into Train and Test data ')
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.3, random_state = 6,stratify = y)

clf = Classifier(train_x = x_train, train_y = y_train, test_x = x_test, test_y = y_test)

train_results = clf.train()
# test_results = clf.test()
confusion_matrix_train = clf.confusion_matrix_train()
confusion_matrix_test = clf.confusion_matrix_test()

print('Exporting the Train Results...')
train_results.to_csv('Train_Results.csv', index = False)
print('Exporting Test Results...')
# test_results.to_csv('Test_Results.csv', index = False)

print('Exporting the Confusion Matrix for Train...')
print(confusion_matrix_train)

print('Exporting the Confusion Matrix for Test...')
print(confusion_matrix_test)