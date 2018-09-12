# test_cntk.py
# CNTK 2.3

import sys
import numpy as np#沒用到
import cntk as C

py_ver = sys.version
cntk_ver = C.__version__
print("Using Python version " + str(py_ver))
print("Using CNTK version " + str(cntk_ver))