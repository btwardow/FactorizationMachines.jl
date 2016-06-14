import numpy as np

from fastFM import sgd
from scipy.sparse import hstack
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import mean_squared_error
from math import sqrt
import sys


train_filename = sys.argv[1]
test_filename = sys.argv[2]

print "Train file:", train_filename
print "Test file :", test_filename

# Load the data
X_train, y_train = load_svmlight_file(train_filename)
n_train = X_train.shape[1]
X_test, y_test = load_svmlight_file(test_filename)
m_test, n_test = X_test.shape
X_test = hstack((X_test, np.zeros((m_test, n_train - n_test), dtype=np.float)))

# Train the model
fm = sgd.FMRegression(n_iter=10, init_stdev=0.01, rank=4,
                      l2_reg_w=0.0, l2_reg_V=0.0, step_size=0.1)
fm.fit(X_train, y_train)

# Make predictions
y_pred = fm.predict(X_test)

# Print RMSE
print(sqrt(mean_squared_error(y_pred, y_test)))
