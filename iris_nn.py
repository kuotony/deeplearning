# iris_nn.py
# CNTK 2.3 with Anaconda 4.1.1 (Python 3.5, NumPy 1.11.1)

# Use a one-hidden layer simple NN with 5 hidden nodes
# iris_train_cntk.txt - 120 items (40 each class)
# iris_test_cntk.txt - remaining 30 items

import numpy as np
import cntk as C

def create_reader(path, input_dim, output_dim, rnd_order, sweeps):
    # rnd_order -> usually True for training
    # sweeps -> usually C.io.INFINITELY_REPEAT for training OR 1 for eval
    x_strm = C.io.StreamDef(field='attribs', shape=input_dim, is_sparse=False)
    y_strm = C.io.StreamDef(field='species', shape=output_dim, is_sparse=False)
    streams = C.io.StreamDefs(x_src=x_strm, y_src=y_strm)
    deserial = C.io.CTFDeserializer(path, streams)
    mb_src = C.io.MinibatchSource(deserial, randomize=rnd_order, max_sweeps=sweeps)
    return mb_src

# ==================================================================================
def main():
    print("\nBegin Iris classification \n")
    print("Using CNTK version = " + str(C.__version__) + "\n")
    
    input_dim = 4
    hidden_dim = 5
    output_dim = 3
    
    train_file = ".\\Data\\iris_train_cntk.txt"
    test_file = ".\\Data\\iris_test_cntk.txt"
    
    # 1. create network
    X = C.ops.input_variable(input_dim, np.float32)
    Y = C.ops.input_variable(output_dim, np.float32)
    print("Creating a 4-5-3 tanh-softmax NN for Iris dataset ")
    with C.layers.default_options(init=C.initializer.uniform(scale=0.01, seed=1)):
        hLayer = C.layers.Dense(hidden_dim, activation=C.ops.tanh, name='hidLayer')(X)
        oLayer = C.layers.Dense(output_dim, activation=None, name='outLayer')(hLayer)
    nnet = oLayer
    model = C.ops.softmax(nnet)
    
    # 2. create learner and trainer
    print("Creating a cross entropy batch=10 SGD LR=0.01 Trainer \n")
    tr_loss = C.cross_entropy_with_softmax(nnet, Y) # not model!
    tr_clas = C.classification_error(nnet, Y)
    max_iter = 2000
    batch_size = 10
    learn_rate = 0.01
    learner = C.sgd(nnet.parameters, learn_rate)
    trainer = C.Trainer(nnet, (tr_loss, tr_clas), [learner])
    
    # 3. create reader for train data
    rdr = create_reader(train_file, input_dim, output_dim, rnd_order=True, sweeps=C.io.INFINITELY_REPEAT)
    iris_input_map = {
        X : rdr.streams.x_src,
        Y : rdr.streams.y_src
    }
    
    # 4. train
    print("Starting training \n")
    for i in range(0, max_iter):
        curr_batch = rdr.next_minibatch(batch_size, input_map=iris_input_map)
        trainer.train_minibatch(curr_batch)
        if i % 500 == 0:
            mcee = trainer.previous_minibatch_loss_average
            macc = (1.0 - trainer.previous_minibatch_evaluation_average) * 100
            print("batch %4d: mean loss = %0.4f, mean accuracy = %0.2f%% " %(i, mcee, macc))
    print("\nTraining complete")
    
    # 5. evaluate model using test data
    print("\nEvaluating test data \n")
    rdr = create_reader(test_file, input_dim, output_dim, rnd_order=False, sweeps=1)
    iris_input_map = {
        X : rdr.streams.x_src,
        Y : rdr.streams.y_src
    }
    num_test = 30
    all_test = rdr.next_minibatch(num_test, input_map=iris_input_map)
    acc = (1.0 - trainer.test_minibatch(all_test)) * 100
    print("Classification accuracy on the 30 test items = %0.2f%%" %acc)
    
    # (could save model here - see text)
    mdl = ".\\Models\\iris_nn.model"
    model.save(mdl, format=C.ModelFormat.CNTKv2)
    
    # 6. use trained model to make prediction
    np.set_printoptions(precision = 1)
    unknown = np.array([[6.4, 3.2, 4.5, 1.5]], dtype=np.float32) # (0 1 0)
    print("\nPredicting Iris species for input features: ")
    print(unknown[0])
    
    # pred_prob = model.eval({X: unknown})
    pred_prob = model.eval(unknown) # simple form works too
    np.set_printoptions(precision = 4, suppress=True)
    print("Prediction probabilities are: ")
    print(pred_prob[0])
    print("\nEnd Iris classification \n ")
# ==================================================================================
if __name__ == "__main__":
    main()