import pandas as pd
import numpy as np
import pickle


class SalesGasNN:
    def __init__(self,X=None,Y=None,layer_one_size=None,alpha=None,iterations=None,m=None,n=None):
        self.X = X
        self.Y =Y
        self.layer_one_size = layer_one_size
        self.alpha = alpha   
        self.iterations  = iterations
        self.m = m
        self.n = n
    def init_params(self):
        np.random.seed(69)
        #layer_one_size = 30
        W1 = np.random.randn(self.layer_one_size, self.n-1)/100 
        b1 = np.random.randn(self.layer_one_size, 1)/100
        #size of second layer fixed at ten to match # features (which is integers 0-9)
        W2 = np.random.randn(2, self.layer_one_size)/100
        b2 = np.random.randn(2, 1)/100
        return W1, b1, W2, b2

    def ReLU(self,Z):
        return np.maximum(Z, 0)

    def softmax(slef,Z):
        ex = np.exp(Z - np.max(Z))
        A = ex/np.sum(ex,axis=0)
        return A

    def forward_prop(self, W1, b1, W2, b2):
        Z1 = W1.dot(self.X) + b1
        A1 = self.ReLU(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def ReLU_deriv(self,Z):
        return Z > 0

    def one_hot(self,Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def backward_prop(self,Z1, A1, Z2, A2, W1, W2):
        m_y = self.Y.size
        one_hot_Y = self.one_hot(self.Y)
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / m_y * dZ2.dot(A1.T)
        db2 = 1 / m_y * np.sum(dZ2)
        dZ1 = W2.T.dot(dZ2) * self.ReLU_deriv(Z1)
        dW1 = 1 / m_y * dZ1.dot(self.X.T)
        db1 = 1 / m_y * np.sum(dZ1)
        return dW1, db1, dW2, db2

    def update_params(self,W1, b1, W2, b2, dW1, db1, dW2, db2):
        W1 = W1 - self.alpha * dW1
        b1 = b1 - self.alpha * db1    
        W2 = W2 - self.alpha * dW2  
        b2 = b2 - self.alpha * db2    
        return W1, b1, W2, b2

    def get_predictions(self,A2):
        return np.argmax(A2, 0)

    def get_accuracy(self,predictions):
        print(predictions, self.Y)
        return np.sum(predictions == self.Y) / self.Y.size

    def gradient_descent(self):
        W1, b1, W2, b2 = self.init_params()
        for i in range(self.iterations):
            Z1, A1, Z2, A2 = self.forward_prop(W1, b1, W2, b2)
            dW1, db1, dW2, db2 = self.backward_prop(Z1, A1, Z2, A2, W1, W2)
            W1, b1, W2, b2 = self.update_params(W1, b1, W2, b2, dW1, db1, dW2, db2)
            if i % 10 == 0:
                print("Iteration: ", i)
                predictions = self.get_predictions(A2)
                print(self.get_accuracy(predictions))
        return W1, b1, W2, b2

    def store_vars(self):
        W1, b1, W2, b2 = self.gradient_descent()
        filePath = 'data.pickle'
        with open(filePath,'wb') as file:
            pickle.dump([W1,b1,W2,b2],file)
        print("data saved succesfully")

if __name__ == '__main__':
    colsToDrop = ['sales_co2','sales_ch4','sales_h2s','sales_btu','index','time']
    df = pd.read_csv('train_data4.csv')
    df = df.dropna()
    Y_train = df['plant_running'].to_numpy().T
    Y_train = np.int32(Y_train)
    data = df.drop(columns=colsToDrop)
    npdf = np.array(data)
    m,n = npdf.shape
    data = data.drop(columns = ['plant_running'])
    print(data)
    data = np.array(data)
    X_train = data.T
    print('Y_train ',Y_train)
    d = SalesGasNN(X_train,Y_train,30,1.0e-5,50,m,n)
    #print(d.__init__())
    print(d.gradient_descent())
    #d.store_vars()