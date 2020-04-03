import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.feature_selection import VarianceThreshold

RANDOM_STATE = 0

class SampleGenerator:
  def __init__(self, dataset, test_size, var_threshold=1e-4):
    self.data = np.loadtxt(dataset, delimiter='\t')
  
    m = len(self.data[1])
    self.X = self.data[:,:m-1]
    self.y = self.data[:,m-1]
    self.labels = np.sort(np.unique(self.y))

    # Remove low-variance features
    self.selector = VarianceThreshold(var_threshold).fit(self.X)
    self.X = self.selector.transform(self.X)

    if test_size > 0:
      # Split into train and test sets
      self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.X,
        self.y,
        test_size=test_size,
        random_state=RANDOM_STATE,
      )
    else:
      # Use all data for training
      self.X_test, self.y_test = np.empty([0, 0]), np.empty([0, 0])
      self.X_train, self.y_train = shuffle(
        self.X,
        self.y,
        random_state=RANDOM_STATE,
      )

  def gen(self):
    print('---Generating data as it comes---')
    return self

class BinaryTestGen(SampleGenerator):
  def __init__(
    self,
    dataset,
    test_size,
    low_pivot=3,
    high_pivot=3,
    var_threshold=1e-4,
  ):
    super().__init__(dataset, test_size, var_threshold)
    self.low_pivot = low_pivot
    self.high_pivot = high_pivot

    # Remove pivots from test
    self.X_test, self.y_test = self.exclude_pivots_(self.X_test, self.y_test)

  def exclude_pivots_(self, X, y):
    X = np.concatenate((X[y<self.low_pivot], X[y>self.high_pivot]))
    y = np.concatenate((y[y<self.low_pivot], y[y>self.high_pivot]))
    # We need to shuffle again
    X, y = shuffle(X, y, random_state=RANDOM_STATE)

    return X, y

  def transform_binary_(self):
    # Labels below low pivot become negative
    self.y_train[self.y_train<self.low_pivot] = 0
    self.y_test[self.y_test<self.low_pivot] = 0
    # Labels above high pivot become positive
    self.y_train[self.y_train>self.high_pivot] = 1
    self.y_test[self.y_test>self.high_pivot] = 1

  def print_binary_size_(self):
    # Print # of positive and negative samples
    print(f'Positives: {len(self.y_train[self.y_train==1]) + len(self.y_test[self.y_test==1])}')
    print(f'Negatives: {len(self.y_train[self.y_train==0]) + len(self.y_test[self.y_test==0])}')
    print(f'Total: {len(self.y_train) + len(self.y_test)}')
    print('--------------------------------------------------------')

  def gen(self):
    print(f'---Generating positive and negative, with pivots {self.low_pivot} and {self.high_pivot}---')
    # Exclude pivots from train
    self.X_train, self.y_train = self.exclude_pivots_(
      self.X_train,
      self.y_train,
    )
    self.transform_binary_()
    self.print_binary_size_()

    return self

class VeryBinaryTestGen(BinaryTestGen):
  def __init__(
    self,
    dataset,
    test_size,
    low_pivot=3,
    high_pivot=3,
    very_low=True,
    very_high=True,
    var_threshold=1e-4,
  ):
    super().__init__(dataset, test_size, low_pivot, high_pivot, var_threshold)
    self.very_low = very_low
    self.very_high = very_high

  def gen(self):
    print(f'---Generating very positive and negative, with pivots {self.low_pivot} and {self.high_pivot}---')
    # Exclude pivots from train
    self.X_train, self.y_train = self.exclude_pivots_(
      self.X_train,
      self.y_train,
    )
    self.make_very_()
    self.transform_binary_()
    self.print_binary_size_()

    return self

  def make_very_(self):
    labels = np.sort(np.unique(self.y_train))
    # Duplicate bottom (very negative) labels
    if self.very_low:
      self.X_train = np.concatenate((
        self.X_train,
        self.X_train[self.y_train==labels[0]],
      ))
      self.y_train = np.concatenate((
        self.y_train,
        self.y_train[self.y_train==labels[0]],
      ))
    # Duplicate top (very positive) labels
    if self.very_high:
      self.X_train = np.concatenate((
        self.X_train,
        self.X_train[self.y_train==labels[labels.size - 1]],
      ))
      self.y_train = np.concatenate((
        self.y_train,
        self.y_train[self.y_train==labels[labels.size - 1]],
      ))
    # We need to shuffle again
    self.X_train, self.y_train = shuffle(
      self.X_train,
      self.y_train,
      random_state=RANDOM_STATE,
    )

class BinaryTestBalancedGen(BinaryTestGen):
  def __init__(
    self,
    dataset,
    test_size,
    low_pivot=3,
    high_pivot=3,
    balance_neg=True,
    balance_pos=True,
    var_threshold=1e-4,
  ):
    super().__init__(dataset, test_size, low_pivot, high_pivot, var_threshold)
    self.balance_neg = balance_neg
    self.balance_pos = balance_pos

  def gen(self):
    print(f'---Generating balanced positive and negative, with pivots {self.low_pivot} and {self.high_pivot}---')
    self.balance_()
    self.transform_binary_()
    self.print_binary_size_()

    return self

  def balance_(self):
    # Make low pivot negative
    if self.balance_neg:
      self.X_train[self.y_train==self.low_pivot] = self.low_pivot - 1
      self.y_train[self.y_train==self.low_pivot] = self.low_pivot - 1
    else:
      self.X_train = self.X_train[self.y_train!=self.low_pivot]
      self.y_train = self.y_train[self.y_train!=self.low_pivot]
    # Make high pivot positive
    if self.balance_pos:
      self.X_train[self.y_train==self.high_pivot] = self.high_pivot + 1
      self.y_train[self.y_train==self.high_pivot] = self.high_pivot + 1
    else:
      self.X_train = self.X_train[self.y_train!=self.high_pivot]
      self.y_train = self.y_train[self.y_train!=self.high_pivot]

class VeryBinaryTestBalancedGen(BinaryTestBalancedGen, VeryBinaryTestGen):
  def __init__(
    self,
    dataset,
    test_size,
    low_pivot=3,
    high_pivot=3,
    balance_neg=True,
    balance_pos=True,
    very_low=True,
    very_high=True,
    var_threshold=1e-4,
  ):
    VeryBinaryTestGen.__init__(
      self,
      dataset,
      test_size,
      low_pivot,
      high_pivot,
      very_low,
      very_high,
      var_threshold,
    )
    BinaryTestBalancedGen.__init__(
      self,
      dataset,
      test_size,
      low_pivot,
      high_pivot,
      balance_neg,
      balance_pos,
      var_threshold,
    )

  def gen(self):
    print(f'---Generating very balanced positive and negative, with pivots {self.low_pivot} and {self.high_pivot}---')
    self.make_very_()
    self.balance_()
    self.transform_binary_()
    self.print_binary_size_()

    return self

#TEST_SIZE = 0.25
#DATASET = 'datasets/dataset_all.txt'
#BinaryTestGen(DATASET, TEST_SIZE, 3, 4).gen()
#VeryBinaryTestGen(DATASET, TEST_SIZE, 3, 4).gen()
#BinaryTestBalancedGen(DATASET, TEST_SIZE, 3, 4, False, True).gen()
#VeryBinaryTestBalancedGen(DATASET, TEST_SIZE, 3, 4, True, True, False, True).gen()
