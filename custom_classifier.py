# left on 5:58
from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)
import random
class ScrappyKNN():
    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
        
    def closest(self,row):
        closest_index=0
        closest_distance = euc(self.X_train[0],row)
        for index in range(1,len(self.X_train)-1):
            distance = euc(self.X_train[index],row)
            if(distance < closest_distance):
                closest_distance = distance
                closest_index=index
        return self.y_train[closest_index]
        
            


import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()

X = iris.data
y = iris.target
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5)
print(y_test)

# from sklearn.neighbors import KNeighborsClassifier
my_classifier = ScrappyKNN()
my_classifier.fit(X_train, y_train)
predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))