"""detect.py —— 窗口级进食检测分类器"""
import sys,os

import pandas as pd

from notebooks.阶段4_baselines_最小可用 import X_test, y_train

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, _HERE)

from build_dataset import build_dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
def train_detector(X,y,test_size=0.3,random_state=42):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=random_state)
    clf=RandomForestClassifier(n_estimators=100,random_state=random_state,n_jobs=-1)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    return clf,acc

if __name__=="__main__":
    X,y=build_dataset()
    clf,acc=train_detector(X,y)
    print("准确率：",acc)

