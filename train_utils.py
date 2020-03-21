import sample_generators as s
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import sklearn.metrics as m

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
      train_1_pr = m.precision_score(y_train, train_pred)
      test_1_pr = m.precision_score(y_test, test_pred)
      train_0_pr = m.precision_score(y_train, train_pred, pos_label=0)
      test_0_pr = m.precision_score(y_test, test_pred, pos_label=0)
      self.f_log.write(f'{name},{train_acc},{test_acc},,{train_1_pr},{test_1_pr},,{train_0_pr},{test_0_pr}\n')
    else:
      print('Train analysis')
      print(m.classification_report(y_train, train_pred))
      print('Test analysis')
      print(m.classification_report(y_test, test_pred))
    
    return clf
  
  def train_cv(self, name, clf, parameters, standardize=False, print_best=False):
    name = f'CV {name}'
    cv = GridSearchCV(clf, parameters, scoring='precision', cv=self.cv)
  
    cv = self.train(name, cv, standardize)
  
    if print_best and not self.log:
      print('Best estimator')
      print(cv.best_estimator_)
      print()

    return cv.best_estimator_

  def get_scaler(self):
    return self.scaler

def print_line():
  print('--------------------------------------------------------')
