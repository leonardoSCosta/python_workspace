import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals.six import StringIO
from sklearn import tree
import pydotplus
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import wget
import os

filename = '/home/leonardo/Python/ML/drug200.csv'

if not(os.path.isfile(filename)):
    filename = wget.download('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv',
                out=filename)

my_data = pd.read_csv(filename)
print(my_data.head(), '\n', my_data.shape)

X = my_data[list(my_data.columns[0:-1])].values
y = my_data[['Drug']].values.tolist()

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW','NORMAL','HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL','HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

print(X[:5], '\n', y[:5])

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
print(drugTree)

drugTree.fit(X_trainset, y_trainset)

predTree = drugTree.predict(X_testset)
print("Predict:", predTree, '\n', "True:", y_testset)

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

acc = [predTree[i] == y_testset[i][0] for i in range(0,len(predTree))]
print(int(sum(acc))/len(acc))

dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')
plt.show()