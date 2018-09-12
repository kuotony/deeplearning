# error_demo.py
# CNTK 2.3, Anaconda 4.1.1

import numpy as np
import cntk as C
targets = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
computed = np.array([[7.0, 2.0, 1.0], [1.0, 9.0, 6.0], [2.0, 1.0, 5.0], [4.0, 7.0, 3.0]], dtype=np.float32)

np.set_printoptions(precision=4, suppress=True)
print("\nTargets = ")
print(targets, "\n")

sm = C.ops.softmax(computed).eval() # apply softmax to computed values
print("\nSoftmax applied to computed = ")
print(sm)

N = len(targets) # 4
n = len(targets[0]) # 3

sum_se = 0.0
for i in range(N): # each item
    for j in range(n):
        err = (targets[i,j] - sm[i,j]) * (targets[i,j] - sm[i,j])
        sum_se += err # accumulate
mean_se = sum_se / N
print("\nMean squared error from scratch = %0.4f" %mean_se)

mean_se = C.losses.squared_error(sm, targets).eval() / 4.0
print("\nMean squared error from CNTK = %0.4f" %mean_se)

sum_cee = 0.0
for i in range(N): # each item
    for j in range(n):
        err = -np.log(sm[i,j]) * targets[i,j]
        sum_cee += err # accumulate
mean_cee = sum_cee / N
print("\nMean cross-entropy error w/ softmax from scratch = %0.4f" %mean_cee)

sum_cee = 0.0
for i in range(N):
    err = C.losses.cross_entropy_with_softmax(computed[i].reshape(1,3), targets[i].reshape(1,3)).eval()
    sum_cee += err
mean_cee = sum_cee / N
print("\nMean cross-entropy error w/ softmax from CNTK = %0.4f" %mean_cee)