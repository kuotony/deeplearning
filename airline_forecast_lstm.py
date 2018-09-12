# airline_forecast_lstm.py
# CNTK v2.3 Ananconda 4.1.1

import numpy as np
import cntk as C
model = C.ops.functions.Function.load(".\\Models\\airline_lstm.model")

np.set_printoptions(precision = 2, suppress=True)
curr = np.array([[5.08, 4.61, 3.90, 4.32]], dtype=np.float32)

for i in range(8):
    pred_change = model.eval(curr)
    print("\nCurrent counts : ", end=""); print(curr)
    print("Forecast change factor = %0.4f" % pred_change[0,0])
    pred_count = curr[0,0] * pred_change[0,0]
    print("Forecast passenger count = %0.2f" % pred_count)
    
    for j in range(3):
        curr[0,j] = curr[0,j+1]
    curr[0,3] = pred_count