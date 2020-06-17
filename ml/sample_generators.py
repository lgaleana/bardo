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

# Generate dataset with labels 0 and 1
class BinaryTestGen(SampleGenerator):
  def __init__(
    self,
    pos_train={4, 5},
    neg_train={1, 2},
  ):
    self.pos_train = pos_train
    self.neg_train = neg_train

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
  def _transform_binary(self, test_size):
    labels = np.unique(self.y_train)
    pos_test = {4, 5}
    neg_test = {1, 2}

    for label in labels:
      if label in self.pos_train or label in self.neg_train:
        self.y_train[self.y_train==label] = int(label in self.pos_train)
      else:
        self.X_train = self.X_train[self.y_train!=label]
        self.y_train = self.y_train[self.y_train!=label]

      if test_size == 0:
        continue

      if label in pos_test or label in neg_test:
        self.y_test[self.y_test==label] = int(label in pos_test)
      else:
        self.X_test = self.X_test[self.y_test!=label]
        self.y_test = self.y_test[self.y_test!=label]

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
    super().gen()
    self._gen_split(test_size)
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
    labels = np.unique(self.y_train)
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
    SampleGenerator.gen(self)
    self._gen_split(test_size)
    self._make_very()
    self._transform_binary(test_size)
#    self.print_size(test_size)

    return self
#
#DATASET = 'data/datasets/lsgaleana-gmail_com_test.txt'
#TEST_SIZE = 0.25
#BinaryTestGen().set(DATASET).gen(TEST_SIZE)
#BinaryTestGen({4, 5, 7}, {1, 2}).set(DATASET).gen(TEST_SIZE)
#BinaryTestGen({4, 5}, {1, 2, 6}).set(DATASET).gen(TEST_SIZE)
#BinaryTestGen({4, 5, 7}, {1, 2, 6}).set(DATASET).gen(TEST_SIZE)
#VeryBinaryTestGen(pos_train={4, 5}).set(DATASET).gen(TEST_SIZE)
