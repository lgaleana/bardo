import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

RANDOM_STATE = 0

class SampleGenerator:
  def set(self, dataset):
    self.dataset = dataset
    return self

  def get_dataset_name(self):
    return self.dataset

  def gen(self):
    self.data = np.loadtxt(self.dataset, delimiter=',')
  
    m = len(self.data[1])
    self.X = self.data[:,:m-1]
    self.y = self.data[:,m-1]

    self.X, self.y = shuffle(
      self.X,
      self.y,
      random_state=RANDOM_STATE,
    )

# Generate dataset with labels 0, 1 and 2
class TernaryTestGen(SampleGenerator):
  def __init__(
    self,
    pos_train={4, 5},
    neg_train={1, 2},
    neu_train={3},
  ):
    self.pos_train = pos_train
    self.neg_train = neg_train
    self.neu_train = neu_train

  # Split into train and test if possible
  def _gen_split(self, test_size):
    if test_size > 0:
      self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.X,
        self.y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=self.y,
      )
    else:
      self.X_train, self.y_train = self.X, self.y

  # Labels are transformed to 0 or 1, or excluded
  def _transform_ternary(self, test_size):
    labels = np.unique(self.y_train)
    pos_test = {4, 5}
    neg_test = {1, 2}
    neu_test = {3}

    for label in labels:
      if label in self.pos_train:
        self.y_train[self.y_train==label] = 1
      elif label in self.neg_train:
        self.y_train[self.y_train==label] = 0
      elif label in self.neu_train:
        self.y_train[self.y_train==label] = 2
      else:
        self.X_train = self.X_train[self.y_train!=label]
        self.y_train = self.y_train[self.y_train!=label]

      if test_size == 0:
        continue

      if label in pos_test:
        self.y_test[self.y_test==label] = 1
      elif label in neg_test:
        self.y_test[self.y_test==label] = 0
      elif label in neu_test:
        self.y_test[self.y_test==label] = 2
      else:
        self.X_test = self.X_test[self.y_test!=label]
        self.y_test = self.y_test[self.y_test!=label]

  def get_name(self):
    name = f'{self.__class__.__name__}'
    negstr = '|'.join([str(i) for i in sorted(list(self.neg_train))])
    posstr = '|'.join([str(i) for i in sorted(list(self.pos_train))])
    neustr = '|'.join([str(i) for i in sorted(list(self.neu_train))])
    name += f' {negstr}+{neustr}+{posstr}'
    return name

  def print_size(self, test_size):
    # Print # of positive and negative samples
    print(f'Positives train: {len(self.y_train[self.y_train==1])}')
    print(f'Negatives train: {len(self.y_train[self.y_train==0])}')
    print(f'Neutrals train: {len(self.y_train[self.y_train==2])}')
    if test_size > 0:
      print(f'Positives test: {len(self.y_test[self.y_test==1])}')
      print(f'Negatives test: {len(self.y_test[self.y_test==0])}')
      print(f'Neutrals test: {len(self.y_test[self.y_test==2])}')
      print(f'Total: {len(self.y_train) + len(self.y_test)}')
    else:
      print(f'Total: {len(self.y_train)}')
    print('--------------------------------------------------------')

  def gen(self, test_size):
#    print(self.get_name())
    super().gen()
    self._gen_split(test_size)
    self._transform_ternary(test_size)
#    self.print_size(test_size)

    return self

# Generate dataset with labels 0 and 1
class BinaryTestGen(TernaryTestGen):
  def __init__(
    self,
    pos_train={4, 5},
    neg_train={1, 2},
  ):
    super().__init__(pos_train, neg_train)

  # Labels are transformed to 0 or 1, or excluded
  def _transform_binary(self, test_size):
    self.X_train = self.X_train[self.y_train!=2]
    self.y_train = self.y_train[self.y_train!=2]

    if test_size > 0:
      self.X_test = self.X_test[self.y_test!=2]
      self.y_test = self.y_test[self.y_test!=2]

  def get_name(self):
    name = f'{self.__class__.__name__}'
    negstr = '|'.join([str(i) for i in sorted(list(self.neg_train))])
    posstr = '|'.join([str(i) for i in sorted(list(self.pos_train))])
    name += f' {negstr}+{posstr}'
    return name

  def print_size(self, test_size):
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
    SampleGenerator.gen(self)
    self._gen_split(test_size)
    self._transform_ternary(test_size)
    self._transform_binary(test_size)
#    self.print_size(test_size)

    return self

class VeryBinaryTestGen(BinaryTestGen):
  def __init__(
    self,
    pos_train={4, 5},
    neg_train={1, 2},
    very=0,
  ):
    super().__init__(
      pos_train,
      neg_train,
    )
    self.very = very

  # Duplicates very positive and negative samples
  def _make_very(self):
    # Duplicate bottom (very negative) labels
    if self.very <= 0:
      self.X_train = np.concatenate((
        self.X_train,
        self.X_train[self.y_train==1],
      ))
      self.y_train = np.concatenate((
        self.y_train,
        self.y_train[self.y_train==1],
      ))
    # Duplicate top (very positive) labels
    if self.very >= 0:
      self.X_train = np.concatenate((
        self.X_train,
        self.X_train[self.y_train==5],
      ))
      self.y_train = np.concatenate((
        self.y_train,
        self.y_train[self.y_train==5],
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
    SampleGenerator.gen(self)
    self._gen_split(test_size)
    self._make_very()
    self._transform_ternary(test_size)
    self._transform_binary(test_size)
#    self.print_size(test_size)

    return self
#
#DATASET = 'data/datasets/lsgaleana-gmail_com_test.txt'
#TEST_SIZE = 0.25
#BinaryTestGen().set(DATASET).gen(TEST_SIZE)
#VeryBinaryTestGen().set(DATASET).gen(TEST_SIZE)
#TernaryTestGen().set(DATASET).gen(TEST_SIZE)
