import sample_generators as s
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import sklearn.metrics as m
from sklearn.model_selection import learning_curve
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

class TrainUtil:
  def __init__(
    self,
    name,
    model,
    data,
    standardize=False,
    cv=False,
    parameters=None,
  ):
    self.name = name
    self.model = model
    self.data = data
    self.standardize = standardize
    self.cv = cv
    self.parameters = parameters

    if self.standardize:
      self.name = f'Scaled {self.name}'
      self.scaler = preprocessing.StandardScaler().fit(self.data.X_train)

  def train(self):
    if self.cv == False:
      self.model = self.train_base_(self.model)
    else:
      self.model = self.train_cv_()

    return self.model

  def train_base_(self, model):
    print(f'---Training {self.name}---')
    if not self.standardize:
      model.fit(self.data.X_train, self.data.y_train)
    else:
      X_train_scaled = self.scaler.transform(self.data.X_train)
      model.fit(X_train_scaled, self.data.y_train)

    return model
  
  def train_cv_(self):
    self.name = f'CV {self.name}'
    gs = GridSearchCV(
      self.model,
      self.parameters,
      scoring=m.make_scorer(
        m.precision_score,
        labels=[1],
        average='macro',
      ),
      cv=self.cv,
    )
  
    gs = self.train_base_(gs)

    return gs.best_estimator_

  def predict(self, X):
    if self.standardize:
      X = self.scaler.transform(X)
    return self.model.predict(X)

  def predict_prod(self, features):
    return self.predict([features])

  def print_metrics(self, log=None):
    train_pred = self.predict(self.data.X_train)
    test_pred = self.predict(self.data.X_test)

    if log is not None:
      print('Writting metrics')
      train_acc = binary_acc(self.data.y_train, train_pred, self.model)
      test_acc = binary_acc(self.data.y_test, test_pred, self.model)
      train_1_pr = m.precision_score(
        self.data.y_train,
        train_pred,
        labels=[1],
        average='macro',
      )
      test_1_pr = m.precision_score(
        self.data.y_test,
        test_pred,
        labels=[1],
        average='macro',
      )
      train_0_pr = m.precision_score(
        self.data.y_train,
        train_pred,
        labels=[0],
        average='macro',
      )
      test_0_pr = m.precision_score(
        self.data.y_test,
        test_pred,
        labels=[0],
        average='macro',
      )
      log.write(f'{self.name},{train_acc},{test_acc},,{train_1_pr},{test_1_pr},,{train_0_pr},{test_0_pr}\n')
    else:
      print('Train analysis')
      print(m.classification_report(self.data.y_train, train_pred))
      print('Test analysis')
      print(m.classification_report(self.data.y_test, test_pred))

  def plot_learning_curve(self, cv):
    print('Plotting learning curve')
    X = self.data.X
    if self.standardize:
      X = self.scaler.transform(self.data.X)

    train_sizes, train_scores, test_scores = learning_curve(
      self.model,
      X,
      self.data.y,
      cv=cv,
      train_sizes=np.linspace(0.1, 1.0, 5),
      scoring=m.make_scorer(binary_acc, model=self.model),
      n_jobs=4,
    )

    plt.plot(
      train_sizes,
      np.mean(train_scores, axis=1),
      label='Train',
    )
    plt.plot(
      train_sizes,
      np.mean(test_scores, axis=1),
      label='Test',
    )

    plt.ylabel('Acc')
    plt.xlabel('Size')
    plt.title(f'{self.name} learning curves')
    plt.legend()
    plt.show()

def binary_acc(y_true, y_pred, model):
  if len(model.classes_) > 2:
    return (m.recall_score(
      y_true,
      y_pred,
      labels=[1],
      average='macro',
    ) + m.recall_score(
      y_true,
      y_pred,
      labels=[0],
      average='macro',
    )) / 2
  else:
    return m.accuracy_score(y_true, y_pred)

def print_line():
  print('--------------------------------------------------------')
