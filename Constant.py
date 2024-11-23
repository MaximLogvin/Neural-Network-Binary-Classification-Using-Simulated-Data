# hiding warnings in tensorflow
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'                           
warnings.filterwarnings('ignore', message='Do not pass an `input_shape`/`input_dim` argument to a layer.')

# actual code
from tensorflow import keras
from keras import layers
from keras.utils import Sequence
from keras.optimizers import Adam,SGD
from keras.layers import Flatten
from keras.callbacks import LearningRateScheduler,Callback
from sklearn.metrics import roc_curve,auc,confusion_matrix,accuracy_score
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import math

# piecewise constant function
def p(x,points):
    for i in range(len(points)-1):
        if(points[i][0]<=x and x<points[i+1][0]):
            return points[i][1]
    if(x==1):
        return 1

# error function
def error(x_grid,y_expected,y_predicted):
    ans=0
    for i in range(len(x_grid)-1):
        ans+=(x_grid[i+1]-x_grid[i])*(y_predicted[i]-y_expected[i])**2
    return ans

# generate n points
def generate_points(n):
    points=[[0,0]]
    for _ in range(n-2):
        points.append([np.random.uniform(0,1),np.random.uniform(0,1)])
    points=sorted(points,key=lambda x: x[0])
    points.append([1,1])
    return points

# create model with specific number of layers and neurons per layer
def create_model(num_layers,num_neurons,optimizer):
   model=keras.models.Sequential()
   model.add(layers.Dense(num_neurons,activation="relu",input_shape=[1]))
   for i in range(1,num_layers):
        model.add(layers.Dense(num_neurons,activation="relu"))
   model.add(layers.Dense(2,activation="softmax"))
   model.compile(optimizer=optimizer,loss="sparse_categorical_crossentropy",metrics=["accuracy"])
   return model


# data generator class
class DataGenerator(Sequence):
    def __init__(self,batch_size,points):                                # output labels
        self.batch_size=batch_size 
        self.points=points
    def __len__(self): 
        return 30000  # number of batches
    
    def __getitem__(self,index):   
        batch_data = np.random.uniform(0,1,self.batch_size)
        batch_labels = np.array([random.choices([0,1],weights=[1-p(x,self.points),p(x,self.points)],k=1)[0] for x in batch_data])
        return batch_data.reshape(-1,1), batch_labels                          # return one batch of data and its labels
    

def bootstrap(errors):
    means=[]
    for sim in range(10000):
        sample=np.random.choice(errors,size=len(errors),replace=True)
        means.append(np.mean(sample))
    return [np.mean(means),np.std(means)]

# write results to file
def write_file(num_layers,num_neurons,error,accuracy,loss):
    with open("constant.txt","a") as f:
        f.write(f"Model with {num_layers} layers, {num_neurons} neurons:  {error},  {accuracy},  {loss}  \n")
        f.write("\n")
        
def write_csv(num_layers,num_neurons,error):
    with open("Constant.csv","a") as res:
            res.write(f"({num_layers},{num_neurons}), {error[0]}")
            res.write(", ")
            res.write("\n")

points=[[0, 0],[0.1, 0.8], [0.2, 0.2],[0.3, 0.7],[0.4, 0.3],[0.5, 0.6],[0.6, 0.4],[0.7, 0.7],[0.8, 0.3],[0.9, 0.8],[1, 1]]

# X = [p[0] for p in points]
# Y = [p[1] for p in points]
# for i in range(len(X)-1):
#     plt.hlines(Y[i],X[i],X[i+1],color="red")
#     plt.scatter(X[i],Y[i],color="black")
#     plt.scatter(X[i+1],Y[i],color="white",edgecolor="black")
# plt.show()

def procedure(num_layers,num_neurons):
    errors=[]
    for _ in range(1):

        print(f"Number of layers: [{num_layers}, {num_neurons}]")
        model=create_model(num_layers,num_neurons,"adam")
                            
        # chunk size
        batch_size=256
        data_gen=DataGenerator(batch_size,points)             
        data_valid=DataGenerator(batch_size,points)           

        # training model 
        history=model.fit(data_gen,validation_data=data_valid)

        # test input
        X_test=np.linspace(0,1,1000).reshape(-1,1)         
        X_shaped=X_test.reshape(-1,1)

        # predicting outputs 
        y_test_prob=model.predict(X_test)     
        y_test_prob=np.array([p[1] for p in y_test_prob]).flatten()          
        y_expected=[p(x,points) for x in X_shaped]

        # error made
        err=error(X_shaped,y_expected,y_test_prob)
        print(err)
        errors.append(err)

        plt.figure(figsize=(8,6))
        plt.plot(X_test,y_test_prob,linewidth=4,color="blue",label="Predicted probability p̂(x)")
        plt.plot(X_test,y_expected,color="red",label="p(x)=x")
        plt.xlabel("Test data")
        plt.ylabel("Probability")
        plt.legend(loc="lower right")
        plt.show()

    errors=np.array(errors).flatten()
    err=bootstrap(errors)
    print(err)
    # write_file(num_layers,num_neurons,err)
    # write_csv(num_layers,num_neurons,err)
    
num_layers_list=[4]
num_neurons_list=[10]
weights=[]
errors=[]
for num_layers in num_layers_list:
    for num_neurons in num_neurons_list:
        procedure(num_layers,num_neurons)
