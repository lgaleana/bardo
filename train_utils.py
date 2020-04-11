import sample_generators as s
from sklearn.model_selection import GridSearchCV, cross_validate, learning_curve
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as m
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

K = 5

class TrainUtil:
  def __init__(
    self,
    name,
    model,
    data,
    standardize=False,
    params=False,
    var_threshold=1e-4
  ):
    self.name = name
    self.model = model
    self.data = data
    self.standardize = standardize
    self.params = params
    self.var_threshold = var_threshold

    if self.standardize:
      self.name = f'Scaled {self.name}'

  def train(self):
    if self.params == False:
      print(f'---Training {self.name}---')
      return self.train_base_(self.model)
    else:
      return self.train_cv_()

  def train_base_(self, model):
    X_train = self.data.X_train
    if self.standardize:
      self.scaler = StandardScaler().fit(self.data.X_train)
      X_train = self.scaler.transform(X_train)
    # Selector of high-variance features
    self.selector = VarianceThreshold(self.var_threshold).fit(self.data.X_train)
    X_train = self.selector.transform(X_train)
    return model.fit(X_train, self.data.y_train)

  def do_cv(self):
    model = self.model
    params = []

    X = self.data.X_train
    y = self.data.y_train
    if self.data.test_size > 0:
      X = np.concatenate((X, self.data.X_test))
      y = np.concatenate((y, self.data.y_test))

    # Adjust params for standardization
    steps = []
    if self.standardize:
      steps.append(('scaler', StandardScaler()))
    steps.append(('selector', VarianceThreshold(self.var_threshold)))
    steps.append(('model', model))
    model = Pipeline(steps)
    if self.params != False:
      for combination in self.params:
        new_comb = {}
        for name, param in combination.items():
          new_comb[f'model__{name}'] = param
        params.append(new_comb)

    # Define cv metrics
    cv_metrics = {
      'acc': 'accuracy',
      'pr': m.make_scorer(
        m.precision_score,
        labels=[1],
        average='macro',
        zero_division=0,
      ),
      'rec': m.make_scorer(
        m.recall_score,
        labels=[1],
        average='macro',
        zero_division=0,
      ),
      'f05': m.make_scorer(f05),
    }

    if self.params == False:
      # Do cross-validation to estimate metrics
      print(f'---CV for {self.name}---')
      results = cross_validate(
        model,
        X,
        y,
        scoring=cv_metrics,
        cv=K,
        return_train_score=True,
        n_jobs=4,
      )
      return {
        'train_acc': np.mean(results['train_acc']),
        'test_acc': np.mean(results['test_acc']),
        'train_pr': np.mean(results['train_pr']),
        'test_pr': np.mean(results['test_pr']),
        'test_rec': np.mean(results['test_rec']),
      }
    else:
      # Do cross-validation to obtain best params
      print(f'---CV search for {self.name}---')
      if 'CV' not in self.name:
        self.name = f'CV {self.name}'
      gs = GridSearchCV(
        model,
        params,
        scoring=cv_metrics,
        refit= 'f05',
        cv=K,
        return_train_score=True,
        n_jobs=4,
      )
      gs.fit(X, y)
      if not self.standardize:
        best_params = gs.best_estimator_.steps[1][1].get_params()
      else:
        best_params = gs.best_estimator_.steps[2][1].get_params()
      return {
        'train_acc': gs.cv_results_['mean_train_acc'][gs.best_index_],
        'test_acc': gs.cv_results_['mean_test_acc'][gs.best_index_],
        'train_pr': gs.cv_results_['mean_train_pr'][gs.best_index_],
        'test_pr': gs.cv_results_['mean_test_pr'][gs.best_index_],
        'test_rec': gs.cv_results_['mean_test_rec'][gs.best_index_],
        'params': best_params,
      }

  def train_cv_(self):
    gs = self.do_cv()
    # Use best params to train a new model with all train data
    print(f'-Training estimator with best params-')
    self.model.set_params(**gs['params'])
    return self.train_base_(self.model)

  def predict(self, X):
    if self.standardize:
      X = self.scaler.transform(X)
    X = self.selector.transform(X)
    return self.model.predict(X)

  def predict_prod(self, features):
    return self.predict([features])[0]

  def get_test_metrics(self):
    train_pred = self.predict(self.data.X_train)
    test_pred = self.predict(self.data.X_test)

    train_acc = m.accuracy_score(self.data.y_train, train_pred)
    test_acc = m.accuracy_score(self.data.y_test, test_pred)
    train_pr = m.precision_score(
      self.data.y_train,
      train_pred,
      labels=[1],
      average='macro',
      zero_division=0,
    )
    test_pr = m.precision_score(
      self.data.y_test,
      test_pred,
      labels=[1],
      average='macro',
      zero_division=0,
    )
    test_rec = m.recall_score(
      self.data.y_test,
      test_pred,
      labels=[1],
      average='macro',
      zero_division=0,
    )

    return train_acc, test_acc, train_pr, test_pr, test_rec

  def get_cv_metrics(self):
    cv = self.do_cv()
    return cv['train_acc'], cv['test_acc'], cv['train_pr'], cv['test_pr'], cv['test_rec']

  def get_name(self):
    return self.name

  def get_params(self):
    return self.model.get_params()

  def plot_learning_curve(self, scorer=None, points=20):
    print('Plotting learning curve')
    # Make X and y similar to train and test transformations
    X = np.concatenate((self.data.X_train, self.data.X_test))
    y = np.concatenate((self.data.y_train, self.data.y_test))
    if self.standardize:
      X = StandardScaler().fit_transform(X)
    X = VarianceThreshold(self.var_threshold).fit_transform(X)

    train_sizes, train_scores, test_scores = learning_curve(
      self.model,
      X,
      y,
      cv=K,
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

def f05(y_true, y_pred):
  pr = m.precision_score(y_true, y_pred, zero_division=0)
  rec = m.recall_score(y_true, y_pred, zero_division=0)
  divisor = ((0.25 * pr) + rec)
  if divisor > 0:
    return 1.25 * (pr * rec) /  divisor
  else:
    return  0

def print_line():
  print('--------------------------------------------------------')
