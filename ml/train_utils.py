import ml.sample_generators as s
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
from copy import deepcopy

N_JOBS = 5

class TrainUtil:
  def __init__(
    self,
    name,
    model,
    datasets,
    test_size,
    standardize=False,
    var_threshold=1e-4,
    k=5,
    params=None,
  ):
    self.name = name
    # Initial config of the model
    self.base_model = model
    # Actual model that was trained
    self.model = None 
    # Dataset generators
    self.data = {
      generator.get_dataset_name(): generator for generator in datasets}
    self.test_size = test_size
    self.standardize = standardize
    # Variance threshold for feature selection
    self.var_threshold = var_threshold
    # Number of folds for cross validation
    self.k = k
    # Param grid to find the best params
    self.params = params
    self.best_params = None

    if self.standardize:
      self.name = f'Scaled {self.name}'
    if self.params is not None:
      self.name = f'CV {self.name}'

  # Generate an instance of training data
  def _gen_train_data(self, test_size):
    generators = list(deepcopy(self.data).values())
    # Generate sampled datasets
    for generator in generators:
      generator.gen(test_size)
    # Generate training data
    X_train, y_train = generators[0].X_train, generators[0].y_train
    for i in range(1, len(generators)):
      X_train = np.concatenate((X_train, generators[i].X_train))
      y_train = np.concatenate((y_train, generators[i].y_train))

    return X_train, y_train

  def train(self):
    print(f'Training {self.name}')
    model = deepcopy(self.base_model)
    if self.params is None:
      self._train_base(model)
    else:
      self._find_best_params()
      # Use best params to train the model
      model.set_params(**self.best_params)
      self._train_base(model)

  def _train_base(self, model):
    # Transform the features
    X_train, y_train = self._gen_train_data(self.test_size)
    if self.standardize:
      self.scaler = StandardScaler().fit(X_train)
      X_train = self.scaler.transform(X_train)
    self.selector = VarianceThreshold(self.var_threshold).fit(X_train)
    X_train = self.selector.transform(X_train)
    # Assign the model that wil be used for future prediction
    self.model = model.fit(X_train, y_train)

  def _find_best_params(self):
    if self.best_params is not None:
      return

    print(f'Finding best params for {self.name}')
    model = deepcopy(self.base_model)
    pipe = self.build_pipeline_(model)
    X_train, y_train = self._gen_train_data(self.test_size)

    # Set up params for pipeline
    params = []
    if self.params is not None:
      for combination in self.params:
        new_comb = {}
        for name, param in combination.items():
          new_comb[f'model__{name}'] = param
        params.append(new_comb)

    # Do cross-validation to obtain best params
    gs = GridSearchCV(
      pipe,
      params,
      scoring=m.make_scorer(f05),
      cv=self.k,
      return_train_score=True,
      n_jobs=N_JOBS,
    )
    gs.fit(X_train, y_train)

    if not self.standardize:
      self.best_params = gs.best_estimator_.steps[1][1].get_params()
    else:
      self.best_params = gs.best_estimator_.steps[2][1].get_params()

  def predict(self, X):
    if self.standardize:
      X = self.scaler.transform(X)
    X = self.selector.transform(X)
    return self.model.predict(X)

  def predict_prod(self, features):
    return self.predict([features])[0]

  def get_test_metrics(self, dataset):
    data = deepcopy(self.data[dataset]).gen(self.test_size)

    train_pred = self.predict(data.X_train)
    test_pred = self.predict(data.X_test)

    train_acc = m.accuracy_score(data.y_train, train_pred)
    test_acc = m.accuracy_score(data.y_test, test_pred)
    train_pr_1 = m.precision_score(
      data.y_train,
      train_pred,
      labels=[1],
      average='macro',
      zero_division=0,
    )
    test_pr_1 = m.precision_score(
      data.y_test,
      test_pred,
      labels=[1],
      average='macro',
      zero_division=0,
    )
    test_rec_1 = m.recall_score(
      data.y_test,
      test_pred,
      labels=[1],
      average='macro',
      zero_division=0,
    )
    train_pr_0 = m.precision_score(
      data.y_train,
      train_pred,
      labels=[0],
      average='macro',
      zero_division=0,
    )
    test_pr_0 = m.precision_score(
      data.y_test,
      test_pred,
      labels=[0],
      average='macro',
      zero_division=0,
    )
    test_rec_0 = m.recall_score(
      data.y_test,
      test_pred,
      labels=[0],
      average='macro',
      zero_division=0,
    )

    return {
      'train_acc': train_acc,
      'test_acc': test_acc,
      'train_pr_1': train_pr_1,
      'test_pr_1': test_pr_1,
      'test_rec_1': test_rec_1,
      'train_pr_0': train_pr_0,
      'test_pr_0': test_pr_0,
      'test_rec_0': test_rec_0,
    }

  def get_cv_metrics(self):
    print(f'CV for {self.name}')
    model = deepcopy(self.base_model)
    if self.params is not None:
      self._find_best_params()
      model.set_params(**self.best_params)
    pipe = self.build_pipeline_(model)
    X_train, y_train = self._gen_train_data(0.0)

    # Cv metrics
    metrics = {
      'acc': 'accuracy',
      'pr_1': m.make_scorer(
        m.precision_score,
        labels=[1],
        average='macro',
        zero_division=0,
      ),
      'rec_1': m.make_scorer(
        m.recall_score,
        labels=[1],
        average='macro',
        zero_division=0,
      ),
      'pr_0': m.make_scorer(
        m.precision_score,
        labels=[0],
        average='macro',
        zero_division=0,
      ),
      'rec_0': m.make_scorer(
        m.recall_score,
        labels=[0],
        average='macro',
        zero_division=0,
      ),
    }

    results = cross_validate(
      pipe,
      X_train,
      y_train,
      scoring=metrics,
      cv=self.k,
      return_train_score=True,
      n_jobs=N_JOBS,
    )

    return {
      'train_acc': np.mean(results['train_acc']),
      'test_acc': np.mean(results['test_acc']),
      'train_pr_1': np.mean(results['train_pr_1']),
      'test_pr_1': np.mean(results['test_pr_1']),
      'test_rec_1': np.mean(results['test_rec_1']),
      'train_pr_0': np.mean(results['train_pr_0']),
      'test_pr_0': np.mean(results['test_pr_0']),
      'test_rec_0': np.mean(results['test_rec_0']),
    }

  def build_pipeline_(self, model):
    # Build model pipeline for CV
    steps = []
    if self.standardize:
      steps.append(('scaler', StandardScaler()))
    steps.append(('selector', VarianceThreshold(self.var_threshold)))
    steps.append(('model', model))
    pipe = Pipeline(steps)

    return pipe

  def get_name(self):
    return self.name

  def get_datasets(self):
    return self.data.values()

  def get_params(self):
    return self.model.get_params()

  def plot_learning_curve(self, scorer=None, points=20):
    print('Plotting learning curve')
    model = deepcopy(self.base_model)
    if self.params is not None:
      self._find_best_params()
      model.set_params(**self.best_params)

    X, y = self._gen_train_data(0.0)
    if self.standardize:
      X = StandardScaler().fit_transform(X)
    X = VarianceThreshold(self.var_threshold).fit_transform(X)

    train_sizes, train_scores, test_scores = learning_curve(
      model,
      X,
      y,
      cv=self.k,
      train_sizes=np.linspace(0.1, 1.0, points),
      scoring=scorer,
      n_jobs=N_JOBS,
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

    plt.ylabel('Score')
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
