import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

RANDOM_STATE = 0

def gen_(test_size):
  data = np.loadtxt('dataset.txt', delimiter=',')

  m = len(data[1])
  X = data[:,:m-1]
  y = data[:,m-1]

  return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)

def exclude_label_(X, y, label):
  return X[y!=label], y[y!=label]

def transform_binary_(train, test, pivot, pivot_class=None):
  train[train<pivot] = 0
  train[train>pivot] = 1
  test[test<pivot] = 0
  test[test>pivot] = 1

  if pivot_class is not None:
    train[train==pivot] = pivot_class
    test[test==pivot] = pivot_class

  return train, test

def gen_binary_(test_size):
  X_train, X_test, y_train, y_test = gen_(test_size)
  # Test will be evaluated excluding 3-star (or neutral) labels
  X_test, y_test = exclude_label_(X_test, y_test, 3)

  return X_train, X_test, y_train, y_test

def print_binary_size(train, test):
  print(f'Positives: {len(train[train==1]) + len(test[test==1])}')
  print(f'Negatives: {len(train[train==0]) + len(test[test==0])}')
  print(f'Total: {len(train) + len(test)}')
  print('--------------------------------------------------------')

def gen_pos_and_neg(test_size):
  print('---Generating only positive and negative samples---')
  X_train, X_test, y_train, y_test = gen_binary_(test_size)

  # Use classes 1 and 2 as positives, and 4 and 5 as negatives
  X_train, y_train = exclude_label_(X_train, y_train, 3)
  y_train, y_test = transform_binary_(y_train, y_test,  3)

  print_binary_size(y_train, y_test)

  return X_train, X_test, y_train, y_test

def gen_very_pos_and_neg(test_size):
  print('---Generating very positive and very negative samples twice (for training)---')
  X_train, X_test, y_train, y_test = gen_binary_(test_size)

  # Duplicate classes 1 and 5
  X_train = np.concatenate((
    X_train[y_train==1],
    X_train[y_train==1],
    X_train[y_train==2],
    X_train[y_train==4],
    X_train[y_train==5],
    X_train[y_train==5],
  ))
  y_train = np.concatenate((
    y_train[y_train==1],
    y_train[y_train==1],
    y_train[y_train==2],
    y_train[y_train==4],
    y_train[y_train==5],
    y_train[y_train==5],
  ))
  # We need to shuffle again
  X_train, y_train = shuffle(X_train, y_train, random_state=RANDOM_STATE)
  y_train, y_test = transform_binary_(y_train, y_test, 3)

  print_binary_size(y_train, y_test)

  return X_train, X_test, y_train, y_test

def gen_pos_and_neg_balanced(test_size):
  print('---Generating balanced positive samples from neutral (for training)---')
  X_train, X_test, y_train, y_test = gen_binary_(test_size)

  y_train, y_test = transform_binary_(y_train, y_test, 3)
  # Delta between negative and positive samples
  delta = len(y_train[y_train==0]) - len(y_train[y_train==1])
  if delta >= 0:
  # Fill in labels 3 as 1
    X_train = np.concatenate((
      X_train[y_train==0],
      X_train[y_train==1],
      X_train[y_train==3][:delta],
    ))
    y_train = np.concatenate((
      y_train[y_train==0],
      y_train[y_train==1],
      y_train[y_train==3][:delta],
    ))
    y_train[y_train==3] = 1
    # We need to shuffle again
    X_train, y_train = shuffle(X_train, y_train, random_state=RANDOM_STATE)
  else:
    # Same as gen_pos_and_neg()
    X_train, y_train = exclude_label_(X_train, y_train, 3)

  print(f'Train Positives: {len(y_train[y_train==1])}')
  print(f'Train Negatives: {len(y_train[y_train==0])}')
  print_binary_size(y_train, y_test)

  return X_train, X_test, y_train, y_test

def gen_pos_and_neutral_neg(test_size):
  print('---Generating positive and negative samples, with neutral samples as negative---')
  X_train, X_test, y_train, y_test = gen_binary_(test_size)

  # Use labels 3 as negative
  y_train, y_test = transform_binary_(y_train, y_test,  3, 0)

  print_binary_size(y_train, y_test)

  return X_train, X_test, y_train, y_test

def gen_pos_neg_and_neutral_train(test_size):
  print('---Generating positive, negative, and neutral (only for training) samples---')
  X_train, X_test, y_train, y_test = gen_binary_(test_size)

  # Only train will have labels 3
  y_train, y_test = transform_binary_(y_train, y_test,  3)

  print(f'Positives: {len(y_train[y_train==1]) + len(y_test[y_test==1])}')
  print(f'Neutral train: {len(y_train[y_train==3])}')
  print(f'Negatives: {len(y_train[y_train==0]) + len(y_test[y_test==0])}')
  print(f'Total: {len(y_train) + len(y_test)}')
  print('--------------------------------------------------------')

  return X_train, X_test, y_train, y_test
#
#
#test_size = 0.25
#gen_pos_and_neg(test_size)
#gen_very_pos_and_neg(test_size)
#gen_pos_and_neg_balanced(test_size)
#gen_pos_and_neutral_neg(test_size)
#gen_pos_neg_and_neutral_train(test_size)
