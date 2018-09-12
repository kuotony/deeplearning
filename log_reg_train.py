# -*- coding: utf-8 -*-
# log_reg_train.py
# logistic regression age-education-sex synthetic data
# CNTK 2.3

import numpy as np
import cntk as C
# ==================================================================================
def main():
    print("\nBegin logistic regression training demo \n")
    ver = C.__version__
    print("Using CNTK version " + str(ver))
    # training data format:
    # 4.0, 3.0, 0
    # 9.0, 5.0, 1
    # . . .
    
    data_file = ".\\age_edu_sex.txt"
    print("Loading data from " + data_file + "\n")
    features_mat = np.loadtxt(data_file, dtype=np.float32, delimiter=",", skiprows=0, usecols=[0,1])
    labels_mat = np.loadtxt(data_file, dtype=np.float32, delimiter=",", skiprows=0, usecols=[2], ndmin=2)
    
    print("Training data: \n")
    combined_mat = np.concatenate((features_mat, labels_mat), axis=1)
    print(combined_mat); print("")
    
    # create model
    features_dim = 2 # x1, x2
    labels_dim = 1   # always 1 for log regression
    
    X = C.ops.input_variable(features_dim, np.float32) # cntk.Variable
    y = C.input_variable(labels_dim, np.float32) # correct class value
    W = C.parameter(shape=(features_dim, 1)) # trainable cntk.Parameter
    b = C.parameter(shape=(labels_dim))
    z = C.times(X, W) + b       # or z = C.plus(C.times(X, W), b)
    p = 1.0 / (1.0 + C.exp(-z)) # or p = C.ops.sigmoid(z)
    model = p # create an alias
    
    # create Learner and Trainer
    ce_error = C.binary_cross_entropy(model, y) # CE a bit more principled for LR
    fixed_lr = 0.010
    learner = C.sgd(model.parameters, fixed_lr)
    trainer = C.Trainer(model, (ce_error), [learner])
    max_iterations = 4000
    
    # train
    print("\nStart training, 4000 iterations, LR = 0.010, mini-batch = 1 \n")
    np.random.seed(4)
    N = len(features_mat)
    for i in range(0, max_iterations):
        row = np.random.choice(N,1) # pick a random row from training items
        trainer.train_minibatch({ X: features_mat[row], y: labels_mat[row] })
        
        if i % 1000 == 0 and i > 0:
            mcee = trainer.previous_minibatch_loss_average
            print(str(i) + " Cross-entropy error on curr item = %0.4f " %mcee)
    
    print("\nTraining complete \n")
    
    np.set_printoptions(precision=4, suppress=True)
    print("Model weights: ")
    print(W.value)
    print("Model bias:")
    print(b.value)
    print("")
    # np.set_printoptions(edgeitems=3,infstr=¡¥inf¡¦,
    # linewidth=75, nanstr=¡¥nan¡¦, precision=8,
    # suppress=False, threshold=1000, formatter=None) # reset all
    
    print("End program ")

# ==================================================================================
if __name__ == "__main__":
    main()