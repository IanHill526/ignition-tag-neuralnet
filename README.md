# ignition-tag-neuralnet
Take in Ignition tags and a MySQL connection and create a neural net that plots predicted and actual data

#primary file is nn_weights_to_pickle
#designed to take mysql connection with ignition historian 
#tables, and create training data from a list of tag names and
#labels by calling the MakeTrainingData
#then the data is trained by calling SalesGasNN with adjustable
#hidden layer size, number of iterations, and learning param alpha.
#the output of SalesGasNN (weights) are stored in a pickle file
#so it only needs to be called once.
