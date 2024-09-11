# ignition-tag-neuralnet
Take in Ignition tags and a MySQL connection and create a neural net that plots predicted and actual data

Primary file is nn_weights_to_pickle.
Designed to take mysql connection with ignition historian.
Tables, and create training data from a list of tag names and
labels by calling the MakeTrainingData class.
Then the data is trained by calling SalesGasNN with adjustable
hidden layer size, number of iterations, and learning param alpha.
The output of SalesGasNN (weights) are stored in a pickle file
so it only needs to be called once.
