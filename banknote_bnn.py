# banknote_bnn.py
# CNTK 2.3 with Anaconda 4.1.1 (Python 3.5, NumPy 1.11.1)

# Use a one-hidden layer simple NN with 10 hidden nodes
# banknote_train_cntk.txt - 80 items (40 authentic, 40 fake)
# banknote_test_cntk.txt - 20 items (10 authentic, 10 fake)
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

# ==================================================================================
def main():
    print("\nBegin banknote binary classification (two-node technique) \n")
    print("Using CNTK version = " + str(C.__version__) + "\n")
    input_dim = 4
    hidden_dim = 10
    output_dim = 2
    
    train_file = ".\\Data\\banknote_train_cntk.txt"
    test_file = ".\\Data\\banknote_test_cntk.txt"
    
    # two-node data files:
    # |stats 4.17110 8.72200 -3.02240 -0.59699 |forgery 0 1 |# authentic
    # |stats -0.20620 9.22070 -3.70440 -6.81030 |forgery 0 1 |# authentic
    # ...
    # |stats 0.60050 1.93270 -3.28880 -0.32415 |forgery 1 0 |# fake
    # |stats 0.91315 3.33770 -4.05570 -1.67410 |forgery 1 0 |# fake
    
    # 1. create network
    X = C.ops.input_variable(input_dim, np.float32)
    Y = C.ops.input_variable(output_dim, np.float32)
    print("Creating a 4-10-2 tanh-softmax NN for partial banknote dataset ")
    with C.layers.default_options(init=C.initializer.uniform(scale=0.01, seed=1)):
        hLayer = C.layers.Dense(hidden_dim, activation=C.ops.tanh, name='hidLayer')(X)
        oLayer = C.layers.Dense(output_dim, activation=None, name='outLayer')(hLayer)
    nnet = oLayer		# train this
    model = C.ops.softmax(nnet)	# predict with this
    
    # 2. create learner and trainer
    print("Creating an ordinary cross entropy batch=10 SGD LR=0.01 Trainer ")
    tr_loss = C.cross_entropy_with_softmax(nnet, Y) # not model!
    tr_clas = C.classification_error(nnet, Y)
    max_iter = 500
    batch_size = 10
    learn_rate = 0.01
    learner = C.sgd(nnet.parameters, learn_rate)
    trainer = C.Trainer(nnet, (tr_loss, tr_clas), [learner])
    
    # 3. create reader for train data
    rdr = create_reader(train_file, input_dim, output_dim, rnd_order=True, sweeps=C.io.INFINITELY_REPEAT)
    banknote_input_map = {
        X : rdr.streams.x_src,
        Y : rdr.streams.y_src
    }
    
    # 4. train
    print("\nStarting training \n")
    for i in range(0, max_iter):
        curr_batch = rdr.next_minibatch(batch_size, input_map=banknote_input_map)
        trainer.train_minibatch(curr_batch)
        if i % 50 == 0:
            mcee = trainer.previous_minibatch_loss_average
            macc = (1.0 - trainer.previous_minibatch_evaluation_average) * 100
            print("batch %4d: mean loss = %0.4f, accuracy = %0.2f%% " %(i, mcee, macc))
    print("\nTraining complete")
    
    # 5. evaluate model using test data
    print("\nEvaluating test data using built-in test_minibatch() \n")
    rdr = create_reader(test_file, input_dim, output_dim, rnd_order=False, sweeps=1)
    banknote_input_map = {
        X : rdr.streams.x_src,
        Y : rdr.streams.y_src
    }
    num_test = 20
    all_test = rdr.next_minibatch(num_test, input_map=banknote_input_map)
    acc = (1.0 - trainer.test_minibatch(all_test)) * 100
    print("Classification accuracy on the 20 test items = %0.2f%%" % acc)
    
    # (could save model here)
    
    # 6. use trained model to make prediction
    np.set_printoptions(precision = 1, suppress=True)
    unknown = np.array([[0.6, 1.9, -3.3, -0.3]], dtype=np.float32) # likely 1 0 = fake
    print("\nPredicting banknote authenticity for input features: ")
    print(unknown[0])
    pred_prob = model.eval({X: unknown})
    np.set_printoptions(precision = 4, suppress=True)
    print("Prediction probabilities are: ")
    print(pred_prob[0])
    if pred_prob[0,0] < pred_prob[0,1]: # maps to (0,1)
        print("Prediction: authentic")
    else: 				# maps to (1,0)
        print("Prediction: fake")
    print("\nEnd banknote classification ")
# ==================================================================================
if __name__ == "__main__":
    main()