# banknote_bnn_onenode.py
# CNTK 2.3 with Anaconda 4.1.1 (Python 3.5, NumPy 1.11.1)

# Use a one-hidden layer simple NN with 10 hidden nodes
# banknote_train_cntk.txt - 80 items (40 authentic, 40 fake)
# banknote_test_cntk.txt - 20 items(10 authentic, 10 fake)

import numpy as np
import cntk as C

def create_reader(path, input_dim, output_dim, rnd_order, sweeps):
    # rnd_order -> usually True for training
    # sweeps -> usually C.io.INFINITELY_REPEAT for training OR 1 for eval
    x_strm = C.io.StreamDef(field='stats', shape=input_dim, is_sparse=False)
    y_strm = C.io.StreamDef(field='forgery', shape=output_dim, is_sparse=False)
    streams = C.io.StreamDefs(x_src=x_strm, y_src=y_strm)
    deserial = C.io.CTFDeserializer(path, streams)
    mb_src = C.io.MinibatchSource(deserial, randomize=rnd_order, max_sweeps=sweeps)
    return mb_src
    
def class_acc(mb, x_var, y_var, model):
    num_correct = 0; num_wrong = 0
    x_mat = mb[x_var].asarray() # batch_size x 1 x features_dim
    y_mat = mb[y_var].asarray() # batch_size x 1 x 1
    for i in range(mb[x_var].shape[0]): # each item in the batch
        p = model.eval(x_mat[i])    # 1 x 1
        y = y_mat[i]                # 1 x 1
        if p[0,0] < 0.5 and y[0,0] == 0.0 or p[0,0] >= 0.5 and y[0,0] == 1.0:
            num_correct += 1
        else:
            num_wrong += 1
    return (num_correct * 100.0) / (num_correct + num_wrong)
# ==================================================================================
def main():
    print("\nBegin banknote binary classification (one-node technique) \n")
    print("Using CNTK version = " + str(C.__version__) + "\n")
    input_dim = 4
    hidden_dim = 10
    output_dim = 1 # NOTE: instead of 2
    train_file = ".\\Data\\banknote_train_cntk_onenode.txt" # NOTE: different file
    test_file = ".\\Data\\banknote_test_cntk_onenode.txt" # NOTE
    # one-node data files:
    # |stats  4.17110 8.72200 -3.02240 -0.59699 |forgery 0 |# authentic
    # |stats -0.20620 9.22070 -3.70440 -6.81030 |forgery 0 |# authentic
    # ...
    # |stats  0.60050 1.93270 -3.28880 -0.32415 |forgery 1 |# fake
    # |stats  0.91315 3.33770 -4.05570 -1.67410 |forgery 1 |# fake
    
    # 1. create network
    X = C.ops.input_variable(input_dim, np.float32)
    Y = C.ops.input_variable(output_dim, np.float32)
    print("Creating a 4-10-1 tanh-logsig NN for partial banknote dataset ")
    with C.layers.default_options(init=C.initializer.uniform(scale=0.01, seed=1)):
        hLayer = C.layers.Dense(hidden_dim, activation=C.ops.tanh, name='hidLayer')(X)
        oLayer = C.layers.Dense(output_dim, activation=C.ops.sigmoid, name='outLayer')(hLayer) # NOTE: sigmoid activation
    model = oLayer # alias
    
    # 2. create learner and trainer
    print("Creating a binary cross entropy batch=10 SGD LR=0.01 Trainer \n")
    tr_loss = C.binary_cross_entropy(model, Y) # NOTE: use model
    # tr_clas = C.classification_error(model, Y) # NOTE: not available for one-node
    max_iter = 1000
    batch_size = 10
    learn_rate = 0.01
    learner = C.sgd(model.parameters, learn_rate) # NOTE: use model
    trainer = C.Trainer(model, (tr_loss), [learner]) # NOTE: no classification error
    
    # 3. create reader for train data
    rdr = create_reader(train_file, input_dim, output_dim, rnd_order=True, sweeps=C.io.INFINITELY_REPEAT)
    banknote_input_map = {
        X : rdr.streams.x_src,
        Y : rdr.streams.y_src
    }
    
    # 4. train
    print("Starting training \n")
    for i in range(0, max_iter):
        curr_batch = rdr.next_minibatch(batch_size, input_map=banknote_input_map)
        trainer.train_minibatch(curr_batch)
        if i % 100 == 0:
            mcee = trainer.previous_minibatch_loss_average # built-in
            ca = class_acc(curr_batch, X, Y, model)        # program-defined
            print("batch %4d: mean loss = %0.4f accuracy = %0.2f%%" %(i, mcee, ca))
    print("\nTraining complete")
    
    # 5. evaluate test data (cannot use trainer.test_minibatch)
    print("\nEvaluating test data using program-defined class_acc() \n")
    rdr = create_reader(test_file, input_dim, output_dim, rnd_order=False, sweeps=1)
    banknote_input_map = {
        X : rdr.streams.x_src,
        Y : rdr.streams.y_src
    }
    num_test = 20
    all_test = rdr.next_minibatch(num_test, input_map=banknote_input_map)
    acc = class_acc(all_test, X, Y, model)
    print("Classification accuracy on the 20 test items = %0.2f%%" % acc)
    
    # (could save model here)
    
    # 6. use trained model to make prediction
    np.set_printoptions(precision = 1, suppress=True)
    unknown = np.array([[0.6, 1.9, -3.3, -0.3]], dtype=np.float32) # likely fake
    print("\nPredicting banknote authenticity for input features: ")
    print(unknown[0])
    pred_prob = model.eval({X: unknown})
    print("Prediction probability is: ")
    print("%0.4f" % pred_prob[0,0])
    if pred_prob[0,0] < 0.5:          # prob(forgery) < 0.5
        print("Prediction: authentic")
    else:
        print("Prediction: fake")
    print("\nEnd banknote classification ")
# ==================================================================================
if __name__ == "__main__":
    main()