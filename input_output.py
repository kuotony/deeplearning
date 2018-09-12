# input_output.py
# demo the NN input-output mechanism
# CNTK 2.3

import numpy as np
import cntk as C

print("\nBegin neural network input-output demo \n")
np.set_printoptions(precision=4, suppress=True, formatter={'float': '{: 0.2f}'.format})

i_node_dim = 4
h_node_dim = 2
o_node_dim = 3

X = C.ops.input_variable(i_node_dim, np.float32)
Y = C.ops.input_variable(o_node_dim, np.float32)

print("Creating a 4-2-3 tanh-softmax neural network")
h = C.layers.Dense(h_node_dim, activation=C.ops.tanh, name='hidLayer')(X)
o = C.layers.Dense(o_node_dim, activation=C.ops.softmax, name='outLayer')(h)
nnet = o

print("\nSetting weights and bias values")
ih_wts = np.array([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06], [0.07, 0.08]], dtype=np.float32)

h_biases = np.array([0.09, 0.10])
ho_wts = np.array([[0.11, 0.12, 0.13], [0.14, 0.15, 0.16]], dtype=np.float32)
o_biases = np.array([0.17, 0.18, 0.19], dtype=np.float32)

h.hidLayer.W.value = ih_wts
h.hidLayer.b.value = h_biases
o.outLayer.W.value = ho_wts
o.outLayer.b.value = o_biases

print("\nSet the input-hidden weights to: ")
print(h.hidLayer.W.value)
print("\nSet the hidden node biases to: ")
print(h.hidLayer.b.value)
print("\nSet the hidden-output weights to: ")
print(o.outLayer.W.value)
print("\nSet the output node biases to: ")
print(o.outLayer.b.value)

print("\nSetting input values to (1.0, 2.0, 3.0, 4.0)")
x_vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
print("\nFeeding input values to hidden layer only ")
h_vals = h.eval({X: x_vals})
print("\nHidden node values:")
print(h_vals)

print("\nFeeding input values to entire network ")
y_vals = nnet.eval({X: x_vals})
print("\nOutput node values:")
print(y_vals)

print("\nEnd input-output demo ")