import pandas as pd
import numpy as np

def data_preprocess_2(df, target):            #normarlize to [-1, 1]
  X_train=df.drop(target,axis=1)
  Y_train=df[target]
  X_label = X_train.columns

  l=np.min(X_train, axis=0)
  X_train=X_train-l

  L=np.max(X_train, axis=0)
  L[L == 0] = 1  # Avoid NaN values
  X_train_norm = X_train/L*2

  X_train_norm=X_train_norm-1

  X_train = pd.DataFrame(X_train_norm, columns=X_label)
  return X_train.to_numpy(), Y_train.to_numpy()

df=pd.read_csv ('data/compas.csv')            ####change the data path
X_train, Y_train=data_preprocess_2(df, 'two_year_recid')

pd.DataFrame(X_train).to_csv('data/X_train.csv', index=False, header=False)
pd.DataFrame(Y_train).to_csv('data/Y_train.csv', index=False, header=False)





