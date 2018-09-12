# boston_reg.py
# CNTK 2.3 with Anaconda 4.1.1 (Python 3.5, NumPy 1.11.1)

# Predict median value of a house in an area near Boston based on area's
# crime rate, age of houses, distance to Boston, pupil-teacher
# ratio, percentage black residents
# boston_train_cntk.txt - 100 items
# boston_test_cntk.txt - 20 items

import numpy as np
import cntk as C

def create_reader(path, input_dim, output_dim, rnd_order, sweeps):
    # rnd_order -> usually True for training
    # sweeps -> usually C.io.INFINITELY_REPEAT for training OR 1 for eval
    x_strm = C.io.StreamDef(field='predictors', shape=input_dim, is_sparse=False)
    y_strm = C.io.StreamDef(field='medval', shape=output_dim, is_sparse=False)
    streams = C.io.StreamDefs(x_src=x_strm, y_src=y_strm)
    # streams = C.variables.Record(x_src=x_strm, y_src=y_strm)
    deserial = C.io.CTFDeserializer(path, streams)
    mb_src = C.io.MinibatchSource(deserial, randomize=rnd_order, max_sweeps=sweeps)
    return mb_src
    
def mb_accuracy(mb, x_var, y_var, model, delta):
    num_correct = 0
    num_wrong = 0
    x_mat = mb[x_var].asarray() # batch_size x 1 x features_dim
    y_mat = mb[y_var].asarray() # batch_size x 1 x 1
    # for i in range(mb[x_var].shape[0]): # each item in the batch
    for i in range(len(mb[x_var])):
        v = model.eval(x_mat[i]) # 1 x 1 predicted value
        y = y_mat[i] 		 # 1 x 1 actual value
        if np.abs(v[0,0] - y[0,0]) < delta: # close enough?
            num_correct += 1
        else:
            num_wrong += 1
    return (num_correct * 100.0) / (num_correct + num_wrong)
# ==================================================================================
def main():
    print("\nBegin median house value regression \n")
    print("Using CNTK version = " + str(C.__version__) + "\n")
    input_dim = 5 # crime, age, distance, pupil-teach, black
    hidden_dim = 20
    output_dim = 1 # median value (x$1000)
    train_file = ".\\Data\\boston_train_cntk.txt"
    test_file = ".\\Data\\boston_test_cntk.txt"
    # data resembles:
    # |predictors 0.041130 33.50 5.40 19.00 396.90 |medval 28.00
    # |predictors 0.068600 62.50 3.50 18.00 393.53 |medval 33.20
    
    # 1. create network
    X = C.ops.input_variable(input_dim, np.float32)
    Y = C.ops.input_variable(output_dim, np.float32)
    print("Creating a 5-20-1 tanh-none regression NN for partial Boston dataset ")
    with C.layers.default_options(init=C.initializer.uniform(scale=0.01, seed=1)):
        hLayer = C.layers.Dense(hidden_dim, activation=C.ops.tanh, name='hidLayer')(X)
        oLayer = C.layers.Dense(output_dim, activation=None, name='outLayer')(hLayer)
    model = C.ops.alias(oLayer) # alias
    
    # 2. create learner and trainer
    print("Creating a squared error batch=5 variable SGD LR=0.02 Trainer \n")
    tr_loss = C.squared_error(model, Y)
    max_iter = 3000
    batch_size = 5
    base_learn_rate = 0.02
    
    sch = C.learning_parameter_schedule([base_learn_rate, base_learn_rate/2], minibatch_size=batch_size, epoch_size=int((max_iter*batch_size)/2))
    learner = C.sgd(model.parameters, sch)
    trainer = C.Trainer(model, (tr_loss), [learner])
    
    # 3. create reader for train data
    rdr = create_reader(train_file, input_dim, output_dim, rnd_order=True, sweeps=C.io.INFINITELY_REPEAT)
    boston_input_map = {
        X : rdr.streams.x_src,
        Y : rdr.streams.y_src
    }
    
    # 4. train
    print("Starting training \n")
    for i in range(0, max_iter):
        curr_batch = rdr.next_minibatch(batch_size, input_map=boston_input_map)
        trainer.train_minibatch(curr_batch)
        if i % int(max_iter/10) == 0:
            mcee = trainer.previous_minibatch_loss_average
            acc = mb_accuracy(curr_batch, X, Y, model, delta=3.00) # program-defined
            print("batch %4d: mean squared error = %8.4f accuracy = %5.2f%%" %(i, mcee, acc))
    print("\nTraining complete")
    
    # 5. evaluate test data (cannot use trainer.test_minibatch)
    print("\nEvaluating test data using program-defined class_acc() \n")
    rdr = create_reader(test_file, input_dim, output_dim, rnd_order=False, sweeps=1)
    boston_input_map = {
        X : rdr.streams.x_src,
        Y : rdr.streams.y_src
    }
    num_test = 20
    all_test = rdr.next_minibatch(num_test, input_map=boston_input_map)
    acc = mb_accuracy(all_test, X, Y, model, delta=3.00)
    print("Prediction accuracy on the 20 test items = %0.2f%%" %acc)
    
    # (could save model here)
    
    # 6. use trained model to make prediction
    np.set_printoptions(precision = 2, suppress=True)
    unknown = np.array([[0.09, 50.00, 4.5, 17.00, 350.00]], dtype=np.float32)
    print("\nPredicting area median home value for feature/predictor values: ")
    print(unknown[0])
    pred_value = model.eval({X: unknown})
    print("\nPredicted home value is: ")
    print("$%0.2f (x1000)" %pred_value[0,0])
    print("\nEnd median house value regression ")
# ==================================================================================
if __name__ == "__main__":
    main()