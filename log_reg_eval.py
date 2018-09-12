# log_reg_eval.py
# logistic regression age-education-sex data
# CNTK 2.3

import numpy as np
# ==================================================================================
def main():
    print("\nBegin logistic regression model evaluation \n")
    
    data_file = ".\\age_edu_sex.txt"
    print("Loading data from " + data_file)
    features_mat = np.loadtxt(data_file, dtype=np.float32, delimiter=",", skiprows=0, usecols=(0,1))
    labels_mat = np.loadtxt(data_file, dtype=np.float32, delimiter=",", skiprows=0, usecols=[2], ndmin=2)
    
    print("Setting weights and bias values \n")
    weights = np.array([-0.2049, 0.9666], dtype=np.float32)
    bias = np.array([-2.2864], dtype=np.float32)
    N = len(features_mat)
    features_dim = 2
    
    print("item  pred_prob  pred_label   act_label   result")
    print("================================================")
    for i in range(0, N): # each item
        x = features_mat[i]
        z = 0.0
        for j in range(0, features_dim): # each feature
            z += x[j] * weights[j]
        z += bias[0]
        pred_prob = 1.0 / (1.0 + np.exp(-z))
        pred_label = 0 if pred_prob < 0.5 else 1
        act_label = labels_mat[i]
        pred_str = 'correct' if np.absolute(pred_label - act_label) < 1.0e-5 else 'WRONG'
        print("%2d %0.4f %0.0f %0.0f %s" %(i, pred_prob, pred_label, act_label, pred_str))
        
    x = np.array([9.5, 4.5], dtype=np.float32)
    print("\nPredicting class for age, education = ")
    print(x)
    z = 0.0
    for j in range(0, features_dim):
        z += x[j] * weights[j]
    z += bias[0]
    p = 1.0 / (1.0 + np.exp(-z))
    print("Predicted p = " + str(p))
    if p < 0.5: print("Predicted class = 0")
    else: print("Predicted class = 1")
        
    print("\nEnd evaluation \n")

# ==================================================================================
if __name__ == "__main__":
    main()