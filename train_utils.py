import sample_generators as s
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as m
from sklearn.model_selection import learning_curve
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

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
      self.scaler = StandardScaler().fit(self.data.X_train)

  def train(self):
    if self.cv == False:
      print(f'---Training {self.name}---')
      return self.train_base_(self.model)
    else:
      return self.train_cv_()

  def train_base_(self, model):
    X_train = self.data.X_train
    if self.standardize:
      X_train = self.scaler.transform(X_train)
    return model.fit(X_train, self.data.y_train)
  
  def train_cv_(self):
    self.name = f'CV {self.name}'
    print(f'---Training {self.name}---')
    model = self.model
    parameters = self.parameters

    # Adjust params for standardization
    if self.standardize:
      model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model),
      ])
      parameters = []
      for combination in self.parameters:
        new_comb = {}
        for name, param in combination.items():
          new_comb[f'model__{name}'] = param
        parameters.append(new_comb)

    # Do cross-validation to obtain best params
    gs = GridSearchCV(
      model,
      parameters,
      scoring=m.make_scorer(
        m.precision_score,
        labels=[1],
        average='macro',
      ),
      cv=self.cv,
    )
    gs.fit(self.data.X_train, self.data.y_train)
    best_params = gs.best_estimator_.get_params()
    if self.standardize:
      best_params = gs.best_estimator_.steps[1][1].get_params()
    print(best_params)

    # Use best params to train a new model with all train data
    print(f'Training selected estimator')
    self.model.set_params(**best_params)
    return self.train_base_(self.model)

  def predict(self, X):
    if self.standardize:
      X = self.scaler.transform(X)
    return self.model.predict(X)

  def predict_prod(self, features):
    return self.predict([features])[0]

  def get_params(self):
    return self.model.get_params()

  def print_metrics(self, log=None):
    train_pred = self.predict(self.data.X_train)
    test_pred = self.predict(self.data.X_test)

    if log is not None:
      print('Writting metrics')
      train_acc = m.accuracy_score(self.data.y_train, train_pred)
      test_acc = m.accuracy_score(self.data.y_test, test_pred)
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
      test_1_rec = m.recall_score(
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
      test_0_rec = m.recall_score(
        self.data.y_test,
        test_pred,
        labels=[0],
        average='macro',
      )
      log.write(f'{self.name},{train_acc},{test_acc},,{train_1_pr},{test_1_pr},{test_1_rec},,{train_0_pr},{test_0_pr},{test_0_rec}\n')
    else:
      print('Train analysis')
      print(m.classification_report(self.data.y_train, train_pred))
      print('Test analysis')
      print(m.classification_report(self.data.y_test, test_pred))

  def plot_learning_curve(self, cv, scorer=None, points=20):
    print('Plotting learning curve')
    # Make X and y similar to train and test transformations
    X = np.concatenate((self.data.X_train, self.data.X_test))
    y = np.concatenate((self.data.y_train, self.data.y_test))
    if self.standardize:
      X = self.scaler.transform(X)

    train_sizes, train_scores, test_scores = learning_curve(
      self.model,
      X,
      y,
      cv=cv,
      train_sizes=np.linspace(0.1, 1.0, points),
      scoring=scorer,
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
    plt.title(f'{self.name}\n{self.data.__class__.__name__} learning curves')
    plt.legend()
    plt.show()

def print_line():
  print('--------------------------------------------------------')
