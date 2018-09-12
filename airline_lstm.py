# airline_lstm.py
# time series regression with a CNTK LSTM
# CNTK 2.3 with Anaconda 4.1.1 (Python 3.5, NumPy 1.11.1)

import numpy as np
import cntk as C
# data resembles:
# |prev 1.12 1.18 1.32 1.29 |next 1.21 |change 1.08035714
# |prev 1.18 1.32 1.29 1.21 |next 1.35 |change 1.14406780

def create_reader(path, input_dim, output_dim, rnd_order, sweeps):
    # rnd_order -> usually True for training
    # sweeps -> usually C.io.INFINITELY_REPEAT Or 1
    x_strm = C.io.StreamDef(field='prev', shape=input_dim, is_sparse=False)
    y_strm = C.io.StreamDef(field='change', shape=output_dim, is_sparse=False)
    z_strm = C.io.StreamDef(field='next', shape=output_dim, is_sparse=False)
    streams = C.io.StreamDefs(x_src=x_strm, y_src=y_strm, z_src=z_strm)
    deserial = C.io.CTFDeserializer(path, streams)
    mb_source = C.io.MinibatchSource(deserial, randomize=rnd_order, max_sweeps=sweeps)
    return mb_source

def mb_accuracy(mb, x_var, y_var, model, delta):
    num_correct = 0
    num_wrong = 0
    x_mat = mb[x_var].asarray() # batch_size x 1 x features_dim
    y_mat = mb[y_var].asarray() # batch_size x 1 x 1
    for i in range(len(mb[x_var])):
        v = model.eval(x_mat[i]) # 1 x 1 predicted change factor
        y = y_mat[i] 		 # 1 x 1 actual change factor
        if np.abs(v[0,0] - y[0,0]) < delta: # close enough?
            num_correct += 1
        else:
            num_wrong += 1
    return (num_correct * 100.0) / (num_correct + num_wrong)
# ==================================================================================
def main():
    print("\nBegin airline passenger time series regression LSTM \n")
    print("Using CNTK version = " + str(C.__version__) + "\n")
    train_file = ".\\Data\\airline_train_cntk.txt"
    
    # 1. create model
    input_dim = 4 # context pattern window
    output_dim = 1 # passenger count increase/decrease factor
    X = C.ops.sequence.input_variable(input_dim) # sequence of 4 items
    Y = C.ops.input_variable(output_dim) # change from X[0]
    Z = C.ops.input_variable(output_dim) # actual next passenger count
    model = None
    with C.layers.default_options():
        model = C.layers.Recurrence(C.layers.LSTM(shape=256))(X)
        model = C.sequence.last(model)
        model = C.layers.Dense(output_dim)(model)
    
    # 2. create the learner and trainer
    learn_rate = 0.01
    tr_loss = C.squared_error(model, Y)
    learner = C.adam(model.parameters, learn_rate, 0.99)
    trainer = C.Trainer(model, (tr_loss), [learner])
    
    # 3. create the training reader; note rnd_order
    rdr = create_reader(train_file, input_dim, output_dim, rnd_order=True, sweeps=C.io.INFINITELY_REPEAT)
    airline_input_map = {
        X : rdr.streams.x_src,
        Y : rdr.streams.y_src
    }
    
    # 4. train
    max_iter = 20000
    batch_size = 10
    print("Starting training \n")
    for i in range(0, max_iter):
        curr_mb = rdr.next_minibatch(batch_size, input_map=airline_input_map)
        trainer.train_minibatch(curr_mb)
        if i % int(max_iter/10) == 0:
            mcee = trainer.previous_minibatch_loss_average
            acc = mb_accuracy(curr_mb, X, Y, model, delta=0.10) # program-defined
            print("batch %6d: mean squared error = %8.4f accuracy = %7.2f%%" %(i, mcee, acc))
    print("\nTraining complete")
    mdl_name = ".\\Models\\airline_lstm.model"
    model.save(mdl_name) # CNTK v2 format is default
    
    # 5. compute model accuracy on data
    print("\nEvaluating LSTM model accuracy on test data with mb_acc() \n")
    rdr = create_reader(train_file, input_dim, output_dim, rnd_order=False, sweeps=1)
    airline_input_map = {
        X : rdr.streams.x_src,
        Y : rdr.streams.y_src
        # no need for Z at this point
    }
    num_items = 140
    all_items = rdr.next_minibatch(num_items, input_map=airline_input_map)
    acc = mb_accuracy(all_items, X, Y, model, delta=0.10)
    print("Prediction accuracy on the 140-item data set = %0.2f%%" % acc)
    
    # 6. save actual-predicted values to make a graph later
    print("\nWriting actual-predicted pct values to file for graphing")
    fout = open(".\\Data\\actual_predicted_lstm.txt", "w")
    rdr = create_reader(train_file, input_dim, output_dim, rnd_order=False, sweeps=1)
    airline_input_map = {
        X : rdr.streams.x_src,
        Y : rdr.streams.y_src,
        Z : rdr.streams.z_src
        # actual next passenger count
    }
    num_items = 140
    all_items = rdr.next_minibatch(num_items, input_map=airline_input_map)
    x_mat = all_items[X].asarray()
    y_mat = all_items[Y].asarray()
    z_mat = all_items[Z].asarray()
    for i in range(all_items[X].shape[0]): # each item in the batch
        v = model.eval(x_mat[i]) # 1 x 1 predicted change
        y = y_mat[i] # 1 x 1 actual change
        z = z_mat[i] # actual next count
        x = x_mat[i] # first item in sequence
        p = x[0,0] * v[0,0] # predicted next count
        fout.write("%0.2f, %0.2f\n" % (z[0,0], p))
    fout.close()
    
    # 7. predict passenger count for Jan. 1961
    np.set_printoptions(precision = 4, suppress=True)
    in_seq = np.array([[5.08, 4.61, 3.90, 4.32]], dtype=np.float32) # last 4 actual
    print("\nForecasting passenger count for Jan. 1961 using: ")
    print(in_seq[0])
    pred_change = model.eval({X: in_seq})
    print("\nForecast change factor is: ")
    print("%0.6f " % pred_change[0,0])
    pred_count = in_seq[0,0] * pred_change[0,0]
    print("Forecast passenger count = %0.4f" % pred_count)
    print("\nEnd demo \n")
# ==================================================================================
if __name__ == "__main__":
    main()