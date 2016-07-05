import sklearn 
from sklearn import tree
features = [[140,1],[130,1],[150,0],[170,0]] # 0 for bumpy and 1 for smooth
labels = [0,0,1,1] # 0 for apple and 1 for orange
clf = tree.DecisionTreeClassifier()
clf.fit(features,labels)
print(clf.predict([150,0]))