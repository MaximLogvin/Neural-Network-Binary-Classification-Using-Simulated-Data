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



def p(x):
    return (np.tanh(8*x-4)-np.tanh(-4))/(np.tanh(4)-np.tanh(-4))
    #return max(0,2*x-1)

# error function
def error(x_grid,y_expected,y_predicted):
    ans=0
    for i in range(len(x_grid)-1):
        ans+=(x_grid[i+1]-x_grid[i])*(y_predicted[i]-y_expected[i])**2
    return ans


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
    def __init__(self,batch_size):                                # output labels
        self.batch_size=batch_size 
    def __len__(self): 
        return 20000  # number of batches
    def __getitem__(self,index):   
        batch_data = np.random.uniform(0,1,self.batch_size)
        batch_labels = np.array([random.choices([0,1],weights=[1-p(x),p(x)],k=1)[0] for x in batch_data])
        return batch_data.reshape(-1,1), batch_labels                          # return one batch of data and its labels
    

# write results to file
def write_file(num_layers,num_neurons,error):
    with open("tanh.txt","a") as f:
        f.write(f"Model with {num_layers} layers, {num_neurons} neurons:  {error}  \n")
        f.write("\n")
def write_csv(num_layers,num_neurons,error):
    with open("Tanh.csv","a") as res:
            res.write(f"({num_layers},{num_neurons}), {error[0]}")
            res.write(", ")
            res.write("\n")

def bootstrap(errors):
    means=[]
    for sim in range(10000):
        sample=np.random.choice(errors,size=len(errors),replace=True)
        means.append(np.mean(sample))
    return [np.mean(means),np.std(means)]

def procedure():
    errors=[]
    for _ in range(1):
        print(f"Number of layers :{num_layers}, {num_neurons}")
        model=create_model(num_layers,num_neurons,"adam")
                            
        # chunk size
        batch_size=256
        data_gen=DataGenerator(batch_size)             
        data_valid=DataGenerator(batch_size)           

        # training model 
        history=model.fit(data_gen,validation_data=data_valid)

        # test input
        X_test=np.linspace(0,1,1000).reshape(-1,1).flatten()         
        X_shaped=X_test.reshape(-1,1).flatten()

        # predicting outputs 
        y_test_prob=model.predict(X_test)     
        y_test_prob=np.array([p[1] for p in y_test_prob]).flatten()          
        y_expected=np.array([p(x) for x in X_shaped],dtype="object").flatten()
        # error made
        err=error(X_shaped,y_expected,y_test_prob)
        print(err)
        errors.append(err)

        plt.figure(figsize=(8,6))
        plt.plot(X_shaped,y_test_prob,linewidth=4,color="blue",label="Predicted probability p̂(x)")
        plt.plot(X_shaped,y_expected,color="red",label="p(x)=x")
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
errors=[]
for num_layers in num_layers_list:
    for num_neurons in num_neurons_list:
        procedure()


               

  

