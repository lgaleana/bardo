import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

RANDOM_STATE = 0

class SampleGenerator:
  def __init__(self, dataset):
    self.data = np.loadtxt(dataset, delimiter='\t')
  
    m = len(self.data[1])
    self.X = self.data[:,:m-1]
    self.y = self.data[:,m-1]

    self.X, self.y = shuffle(
      self.X,
      self.y,
      random_state=RANDOM_STATE,
    )

class BinaryTestGen(SampleGenerator):
  def __init__(
    self,
    dataset,
    pivot=3,
    balance=0,
  ):
    super().__init__(dataset)
    self.pivot = pivot
    self.balance = balance

  def gen_split_(self, test_size):
    if test_size > 0:
      # Split into train and test sets
      self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.X,
        self.y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=self.y,
      )
      # Remove pivot from test
      self.X_test = self.X_test[self.y_test!=self.pivot]
      self.y_test = self.y_test[self.y_test!=self.pivot]
    else:
      # Use all data for training
      self.X_train, self.y_train = self.X, self.y

  def balance_(self):
    # Make pivot negative
    if self.balance < 0:
      self.y_train[self.y_train==self.pivot] = self.pivot - 1
    # Make pivot positive
    elif self.balance > 0:
      self.y_train[self.y_train==self.pivot] = self.pivot + 1
    # Remove pivot
    else:
      self.X_train = self.X_train[self.y_train!=self.pivot]
      self.y_train = self.y_train[self.y_train!=self.pivot]

  def transform_binary_(self, test_size):
    # Labels below pivot become negative; above pivot, positive
    self.y_train[self.y_train<self.pivot] = 0
    self.y_train[self.y_train>self.pivot] = 1
    if test_size > 0:
      self.y_test[self.y_test<self.pivot] = 0
      self.y_test[self.y_test>self.pivot] = 1

  def get_name(self):
    name = f'{self.__class__.__name__}'
    if self.balance < 0:
      name += f' balanced to low'
    elif self.balance > 0:
      name += f' balanced to high'
    return name

  def print_binary_size_(self, test_size):
    # Print # of positive and negative samples
    print(f'Positives train: {len(self.y_train[self.y_train==1])}')
    print(f'Negatives train: {len(self.y_train[self.y_train==0])}')
    if test_size > 0:
      print(f'Positives test: {len(self.y_test[self.y_test==1])}')
      print(f'Negatives test: {len(self.y_test[self.y_test==0])}')
      print(f'Total: {len(self.y_train) + len(self.y_test)}')
    else:
      print(f'Total: {len(self.y_train)}')
    print('--------------------------------------------------------')

  def gen(self, test_size):
#    print(self.get_name())
    self.gen_split_(test_size)
    self.balance_()
    self.transform_binary_(test_size)
#    self.print_binary_size_(test_size)

    return self

class VeryBinaryTestGen(BinaryTestGen):
  def __init__(
    self,
    dataset,
    pivot=3,
    balance=0,
    very=0,
  ):
    BinaryTestGen.__init__(
      self,
      dataset,
      pivot,
      balance,
    )
    self.very = very

  def make_very_(self):
    labels = np.sort(np.unique(self.y_train))
    # Duplicate bottom (very negative) labels
    if self.very <= 0:
      self.X_train = np.concatenate((
        self.X_train,
        self.X_train[self.y_train==labels[0]],
      ))
      self.y_train = np.concatenate((
        self.y_train,
        self.y_train[self.y_train==labels[0]],
      ))
    # Duplicate top (very positive) labels
    if self.very >= 0:
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

  def get_name(self):
    name = ''
    if self.very < 0:
      name += 'Bottom '
    elif self.very > 0:
      name += 'Top '
    return f'{name}{super().get_name()}'

  def gen(self, test_size):
#    print(self.get_name())
    self.gen_split_(test_size)
    self.make_very_()
    self.balance_()
    self.transform_binary_(test_size)
#    self.print_binary_size_(test_size)

    return self
#
#DATASET = 'datasets/dataset_test6.txt'
#TEST_SIZE = 0.25
#BinaryTestGen(DATASET).gen(TEST_SIZE),
#VeryBinaryTestGen(DATASET).gen(TEST_SIZE),
#BinaryTestGen(DATASET, 3, -1).gen(TEST_SIZE),
#VeryBinaryTestGen(DATASET, 3, 0, 1).gen(TEST_SIZE),
#VeryBinaryTestGen(DATASET, 3, -1, 1).gen(TEST_SIZE),
#VeryBinaryTestGen(DATASET, 3, -1).gen(TEST_SIZE),
