import numpy as np
from sklearn.neural_network import MLPClassifier as mlp

def mlp_wrapper(train_data, train_labels, layers, neurons, activation, solver, alpha):
      
    model = mlp(solver=solver, activation=activation, alpha=alpha, hidden_layer_sizes=(layers, neurons), random_state=1, max_iter=500)
    model.fit(train_data, train_labels)
    
    return model