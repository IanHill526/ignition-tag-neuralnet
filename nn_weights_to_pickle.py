from MakeTrainingData import MakeTrainingData
from salesgasneuralnet import SalesGasNN
import mysql.connector
import pandas as pd
import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt

def df(db,tableName,labels,tagPaths):
    d = MakeTrainingData(db,tableName,labels,tagPaths)
    df = d.dataFrame().dropna()
    return df

def addTime(df):
    running  = list(df['plant_running'])
    count = 0
    minutesSinceLastDowntime = []
    for j in range(len(running)):
        if running[j] == 1:
            count+=1
        else:
            count = 0
        minutesSinceLastDowntime.append(count)
    df['time'] = minutesSinceLastDowntime
    return df

def merge_dfs(dflist):
    #dflist is a list of dataframes eg [df1,df2,...]
    return pd.concat(dflist,ignore_index=True)

def dropCols(colsToDrop,df):
    #df = df.reset_index()
    #df = df.drop(['index'],axis=1)
    mainIndex = str(list(df.columns)[0])
    df.set_index(mainIndex,inplace=True)
    print(df)
    return df.drop(columns = colsToDrop)

def updateDfCsv(filename,df):
    path = os.path.dirname(os.path.realpath(sys.argv[0])) +'\&' + filename
    if os.path.isfile(path):
        os.remove(path)
    df.to_csv(filename)


def prep_nn_data(df):
    Y = df['plant_running'].to_numpy().T
    Y = np.int32(Y)
    data = dropCols(colsToDrop,df)
    npdf = np.array(data)
    m,n = npdf.shape
    data = data.drop(columns = ['plant_running'])
    print(data)
    data = np.array(data)
    X = data.T
    return X,Y,m,n


def trainNNToPickle(X,Y,m,n,layerOneSize,alpha,nIter):
    d = SalesGasNN(X,Y,layerOneSize,alpha,nIter,m,n) 
    d.store_vars()


    
def getNNOutput(): #take a look at the weights
    file = open('data.pickle','rb')
    response = pickle.load(file)
    W1,b1,W2,b2 = response
    file.close()
    return W1,b1,W2,b2
    

# db= mysql.connector.connect(
#     host="",
#     user="",
#     password="",
#     database = ''
# ) 

# tableName = ''
#labels = [] #list of labels
#tagPaths = [] #list of tag paths
colsToDrop = ['time']
layerOneSize = 10
alpha = 1.0e-5
nIter = 600


W1,b1,W2,b2 = getNNOutput()

df_test = df(db,tableName,labels,tagPaths)
df_test = addTime(df_test)
updateDfCsv('test.csv',df_test)
X_test,Y_test,m_test,n_test = prep_nn_data(df_test)
d = SalesGasNN(X_test,Y_test,layerOneSize,alpha,nIter,m_test,n_test)
na,na,na, A2 = d.forward_prop(W1, b1, W2, b2)
h_x = list(A2[1])
df_test = df_test.reset_index()
df_test.index.name = 'index'
t = list(df_test.index.values)
print(type(t),t)
print(type(h_x))
print(h_x)
def makePlots():
    fig,ax = plt.subplots()
    ax.plot(t,Y_test,'r.',label = 'actual data')
    ax.plot(t,h_x,'g.', label  = 'predicted data',markersize =1,alpha = 0.9)
    ax.legend(['Actual Data','Predicted Data'])
    plt.xlabel('time (minutes into March 2024)')
    plt.title('Predicted & Actual Output of SALES_GAS_OK')
    plt.show()

makePlots()