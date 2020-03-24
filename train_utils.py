import sample_generators as s
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import sklearn.metrics as m
from sklearn.model_selection import learning_curve
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

class TrainUtil:
  def __init__(self, generator, test_size, cv=6, log=False):
    self.X_train, self.X_test, self.y_train, self.y_test = generator(test_size)
    self.cv = cv
    self.scaler = preprocessing.StandardScaler().fit(self.X_train)
    self.log = log

    if log:
      self.f_log = open(f'reports/{generator.__name__}_{len(self.y_train)}_{len(self.y_test)}_{cv}.txt', 'a+')
      self.f_log.write(f'{generator.__name__}\n')
      self.f_log.write(',Train Acc,Test Acc,,Train 1 Pr,Test 1 Pr,,Train 0 Pr,Test 0 Pr\n')

  def train(self, name, clf, standardize=False):
    y_train, y_test = self.y_train, self.y_test
    if not standardize:
      X_train, X_test = self.X_train, self.X_test
    if standardize:
      name = f'Scaled {name}'
      X_train = self.scaler.transform(self.X_train)
      X_test = self.scaler.transform(self.X_test)
  
    print(f'---Training {name}---')
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
  
    if self.log:
      train_acc = m.accuracy_score(y_train, train_pred)
      test_acc = m.accuracy_score(y_test, test_pred)
      train_1_pr = m.precision_score(y_train, train_pred, labels=[1], average='macro')
      test_1_pr = m.precision_score(y_test, test_pred, labels=[1], average='macro')
      train_0_pr = m.precision_score(y_train, train_pred, labels=[0], average='macro')
      test_0_pr = m.precision_score(y_test, test_pred, labels=[0], average='macro')
      self.f_log.write(f'{name},{train_acc},{test_acc},,{train_1_pr},{test_1_pr},,{train_0_pr},{test_0_pr}\n')
    else:
      print('Train analysis')
      print(m.classification_report(y_train, train_pred))
      print('Test analysis')
      print(m.classification_report(y_test, test_pred))

    self.name = name
    self.clf = clf
    
    return clf
  
  def train_cv(self, name, clf, parameters, standardize=False, print_best=False):
    name = f'CV {name}'
    cv = GridSearchCV(
      clf,
      parameters,
      scoring=m.make_scorer(m.precision_score, labels=[1], average='macro'),
      cv=self.cv,
    )
  
    cv = self.train(name, cv, standardize)
  
    if print_best:
      print('Best estimator')
      print(cv.best_estimator_)
      print()

    self.clf = cv.best_estimator_

    return cv.best_estimator_

  def get_scaler(self):
    return self.scaler

  def plot_learning_curve(self, standardize=False):
    print('---Plotting learning curve---')
    X = np.concatenate((self.X_train, self.X_test))
    y = np.concatenate((self.y_train, self.y_test))
    X, y = shuffle(X, y, random_state=0)
    if standardize:
      scaler = preprocessing.StandardScaler().fit(X)
      X = scaler.transform(X)

    train_sizes, train_scores, test_scores = learning_curve(
      self.clf,
      X,
      y,
      cv=self.cv,
      train_sizes=np.linspace(0.1, 1.0, 5),
      scoring='precision',
      n_jobs=4,
    )

    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Train')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Test')

    plt.ylabel('Precision')
    plt.xlabel('Size')
    plt.title(f'{self.name} learning curves')
    plt.legend()
    plt.show()

def print_line():
  print('--------------------------------------------------------')
