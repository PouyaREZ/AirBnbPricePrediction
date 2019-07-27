from sklearn import feature_selection
import numpy as np
import pandas as pd
import math


X_train = pd.read_csv('../Data/data_cleaned_train_comments_X.csv')
y_train = pd.read_csv('../Data/data_cleaned_train_y.csv')

X_val = pd.read_csv('../Data/data_cleaned_val_comments_X.csv')
y_val = pd.read_csv('../Data/data_cleaned_val_y.csv')

F_vals, p_vals = feature_selection.f_regression(X_train, y_train)
def clean_pvals(entry):
    if (math.isnan(entry)):
        return 100
    else:
        return entry
clean_pvals_vectorized = np.vectorize(clean_pvals)
p_vals = clean_pvals_vectorized(p_vals)
print(p_vals.shape)
THRESH = 1e-20
print(p_vals[p_vals<THRESH].shape)
print(X_train.columns[p_vals<THRESH])
print(list(X_train.columns[p_vals<THRESH]))
print(len(list(X_train.columns[p_vals<THRESH])))
np.save('../Data/selected_coefs_pvals.npy', X_train.columns[p_vals<THRESH])
