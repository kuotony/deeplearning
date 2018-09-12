# iris_load.py
# CNTK 2.3

import numpy as np
import cntk as C
print("Loading saved Iris model")
model = C.ops.functions.Function.load(".\\Models\\iris_nn.model")

np.set_printoptions(precision = 1, suppress=True)
unknown = np.array([[6.4, 3.2, 4.5, 1.5]], dtype=np.float32) # (0 1 0)
print("\nPredicting Iris species for input features: ")
print(unknown[0])

pred_prob = model.eval(unknown)
np.set_printoptions(precision = 4, suppress=True)
print("Prediction probabilities are: ")
print(pred_prob[0])

print("\nDone \n")