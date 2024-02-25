import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import enum
import sys
from numpy.linalg import norm
import statistics as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,multilabel_confusion_matrix
from sklearn.model_selection import train_test_split


def cosine(test_point,train_set):
    cosine_lambda = lambda train_point: 1-(np.dot(test_point,train_point) / (norm(test_point)*norm(train_point)))
    return np.argsort(np.array(list(map(cosine_lambda,train_set))))

def euclidean(test_point,train_set):
    euclidean_lambda = lambda train_point: norm(test_point-train_point)
    return np.argsort(np.array(list(map(euclidean_lambda,train_set))))

def manhattan(test_point,train_set):
    manhattan_lambda = lambda train_point: np.sum(np.abs(test_point-train_point))
    return np.argsort(np.array(list(map(manhattan_lambda,train_set))))

distance_func = [cosine,euclidean,manhattan]

class encoders(enum.Enum):
    ResNet = 1
    VIT = 2

class distances(enum.Enum):
    cosine = 1
    euclidean = 2
    manhattan = 3

class knn:
    def __init__(self,k,encoder,distance):
        self.hyperparameters={
            'k':k,
            'encoder':encoder,
            'distance':distance
        }
        
    def fit(self,x_train,y_train):
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
    
    def test(self,x_test,y_test):
        test_result = np.vectorize(distance_func[self.hyperparameters['distance']-1],excluded=['train_set'],otypes=[np.ndarray])
        
        y_predicted = np.array(list(map(lambda dist: st.mode(self.y_train[dist][:self.hyperparameters['k']]),np.array(test_result(test_point = x_test,train_set = self.x_train)))))
        
        return (accuracy_score(np.array(y_test),y_predicted),f1_score(y_predicted,np.array(y_test),average='weighted',zero_division=1.0),precision_score(y_predicted,np.array(y_test),average='weighted',zero_division=1.0),recall_score(y_predicted,np.array(y_test),average='weighted',zero_division=1.0))

data = sys.argv[1]
k_val = int(sys.argv[2])
encoder = int(sys.argv[3])
dis_metric = int(sys.argv[4])


df = np.load(data,allow_pickle=True)
df = pd.DataFrame(df,columns=['game_id','ebd_ResNet','ebd_VIT','Label','TimeStamps'])

labels = np.array(df.iloc[:,3])

labels,frequency = np.unique(labels,return_counts = True)
df.iloc[:,1] = df.iloc[:,1].map(lambda cell: np.array(cell).flatten(),na_action='ignore')
df.iloc[:,2] = df.iloc[:,2].map(lambda cell: np.array(cell).flatten(),na_action='ignore')

x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,encoder],df.loc[:,'Label'] , random_state=42,test_size=0.25, shuffle=True)

classifier = knn(k=k_val,encoder=encoder,distance=dis_metric)
classifier.fit(x_train,y_train)
accuracy,f1,precision,recall = classifier.test(x_test,y_test)

print("Accuracy =",accuracy/5)
print("f1 =",f1/5)
print("Precision =", precision/5)
print("Recall =", recall/5)