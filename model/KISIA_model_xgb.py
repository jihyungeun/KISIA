import xgboost as xgb
import numpy as np
import pandas as pd
import sys
import os
import time
#from sklearn.model_selection import test_split
from sklearn.cross_validation import test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def read_data():

    data = pd.read_csv('대충 뽑은 데이터csv파일 이름')

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y

def split_data(X, y):
    X_info, X_test, y_info, y_test = test_split(X, y, test_size=0.2, random_state=0)
    return X_info, X_test, y_info, y_test

def first_model(X_info, y_info):
    model = xgb.XGBClassifier()
    model.fit(X_info, y_info)
    return model

def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: %.2f%%' % (accuracy * 100.0))

    print('Classification report:')
    print(classification_report(y_test, y_pred))

    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))

    print('ROC curve:')
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='XGBoost')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('XGBoost ROC Curve')
    plt.show()

    auc = roc_auc_score(y_test, y_pred_prob)
    print('AUC: %.2f%%' % (auc * 100.0))

    model.save_model('model.bin')
    print('Model saved')

    model = xgb.Booster()
    model.load_model('model.bin')
    print('Model loaded')

def main():
    X, y = read_data()
    X_info, X_test, y_info, y_test = split_data(X, y)
    model = first_model(X_info, y_info)
    test_model(model, X_test, y_test)
    