import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

RANDOM_STATE = 0

class SampleGen:
  def __init__(self, dataset, test_size):
    self.data = np.loadtxt(dataset, delimiter=',')
  
    m = len(self.data[1])
    self.X = self.data[:,:m-1]
    self.y = self.data[:,m-1]

    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
      self.X,
      self.y,
      test_size=test_size,
      random_state=RANDOM_STATE,
    )
  
  def exclude_label_(self, X, y, label):
    return X[y!=label], y[y!=label]
  
  def transform_binary_(self, train, test, pivot, pivot_class=None):
    train[train<pivot] = 0
    train[train>pivot] = 1
    test[test<pivot] = 0
    test[test>pivot] = 1
  
    if pivot_class is not None:
      train[train==pivot] = pivot_class
      test[test==pivot] = pivot_class
  
    return train, test
  
  def print_binary_size_(self):
    print(f'Positives: {len(self.y_train[self.y_train==1]) + len(self.y_test[self.y_test==1])}')
    print(f'Negatives: {len(self.y_train[self.y_train==0]) + len(self.y_test[self.y_test==0])}')
    print(f'Total: {len(self.y_train) + len(self.y_test)}')
    print('--------------------------------------------------------')

  def make_test_binary_(self):
    # Exclude label 3 from test sets
    self.X_test, self.y_test = self.exclude_label_(
      self.X_test,
      self.y_test,
      3,
    )

  def gen(self):
    print('---Generating data as it comes---')
    return self

class PosAndNegGen(SampleGen):
  def gen(self):
    print('---Generating only positive and negative samples---')
    self.make_test_binary_()
  
    # Use classes 1 and 2 as positives, and 4 and 5 as negatives
    self.X_train, self.y_train = self.exclude_label_(
      self.X_train,
      self.y_train,
      3,
    )
    self.y_train, self.y_test = self.transform_binary_(
      self.y_train,
      self.y_test,
      3,
    )
  
    self.print_binary_size_()

    return self
  
class VeryPosAndNegGen(SampleGen):
  def gen(self):
    print('---Generating very positive and very negative samples twice (for training)---')
    self.make_test_binary_()
  
    # Duplicate classes 1 and 5
    self.X_train = np.concatenate((
      self.X_train[self.y_train==1],
      self.X_train[self.y_train==1],
      self.X_train[self.y_train==2],
      self.X_train[self.y_train==4],
      self.X_train[self.y_train==5],
      self.X_train[self.y_train==5],
    ))
    self.y_train = np.concatenate((
      self.y_train[self.y_train==1],
      self.y_train[self.y_train==1],
      self.y_train[self.y_train==2],
      self.y_train[self.y_train==4],
      self.y_train[self.y_train==5],
      self.y_train[self.y_train==5],
    ))
    # We need to shuffle again
    self.X_train, self.y_train = shuffle(
      self.X_train,
      self.y_train,
      random_state=RANDOM_STATE,
    )
    self.y_train, self.y_test = self.transform_binary_(
      self.y_train,
      self.y_test,
      3,
    )
  
    self.print_binary_size_()

    return self
  
class PosAndNegBalancedGen(SampleGen):
  def gen(self):
    print('---Generating balanced positive samples from neutral (for training)---')
    self.make_test_binary_()
  
    self.y_train, self.y_test = self.transform_binary_(
      self.y_train,
      self.y_test,
      3,
    )
    # Delta between negative and positive samples
    delta = len(self.y_train[self.y_train==0]) - len(self.y_train[self.y_train==1])
    if delta >= 0:
    # Fill in labels 3 as 1
      self.X_train = np.concatenate((
        self.X_train[self.y_train==0],
        self.X_train[self.y_train==1],
        self.X_train[self.y_train==3][:delta],
      ))
      self.y_train = np.concatenate((
        self.y_train[self.y_train==0],
        self.y_train[self.y_train==1],
        self.y_train[self.y_train==3][:delta],
      ))
      self.y_train[self.y_train==3] = 1
      # We need to shuffle again
      self.X_train, self.y_train = shuffle(
        self.X_train,
        self.y_train,
        random_state=RANDOM_STATE,
      )
    else:
      # Same as gen_pos_and_neg()
      self.X_train, self.y_train = self.exclude_label_(
        self.X_train,
        self.y_train,
        3,
      )
  
    print(f'Train Positives: {len(self.y_train[self.y_train==1])}')
    print(f'Train Negatives: {len(self.y_train[self.y_train==0])}')
    self.print_binary_size_()

    return self
  
class PosAndNeutralNegGen(SampleGen):
  def gen(self):
    print('---Generating positive and negative samples, with neutral samples as negative---')
    self.make_test_binary_()
  
    # Use labels 3 as negative
    self.y_train, self.y_test = self.transform_binary_(
      self.y_train,
      self.y_test,
      3, 
      0,
    )
  
    self.print_binary_size_()

    return self
  
class PosNegAndNeutralTrainGen(SampleGen):
  def gen(self):
    print('---Generating positive, negative, and neutral (only for training) samples---')
    self.make_test_binary_()
  
    # Only train will have labels 3
    self.y_train, self.y_test = self.transform_binary_(
      self.y_train,
      self.y_test,
      3,
    )
  
    print(f'Positives: {len(self.y_train[self.y_train==1]) + len(self.y_test[self.y_test==1])}')
    print(f'Neutral train: {len(self.y_train[self.y_train==3])}')
    print(f'Negatives: {len(self.y_train[self.y_train==0]) + len(self.y_test[self.y_test==0])}')
    print(f'Total: {len(self.y_train) + len(self.y_test)}')
    print('--------------------------------------------------------')

    return self
#
#
#TEST_SIZE = 0.25
#DATASET = 'dataset.txt'
#PosAndNegGen(DATASET, TEST_SIZE).gen()
#VeryPosAndNegGen(DATASET, TEST_SIZE).gen()
#PosAndNegBalancedGen(DATASET, TEST_SIZE).gen()
#PosAndNeutralNegGen(DATASET, TEST_SIZE).gen()
#PosNegAndNeutralTrainGen(DATASET, TEST_SIZE).gen()
