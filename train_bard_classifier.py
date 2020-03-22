from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import sample_generators as s
from train_utils import TrainUtil, print_line
from math import ceil
#from sklearn.externals import joblib


# Generators generate different training samples
# We want to test nmany
generators = [
  s.gen_pos_and_neg,
  s.gen_very_pos_and_neg,
  s.gen_pos_and_neg_balanced,
  s.gen_pos_and_neutral_neg,
]

for generator in generators:
  t = TrainUtil(generator, 0.25, 6, True)
  # Linear SVC
  name = 'Linear SVC'
  svc = LinearSVC(dual=False)
  parameters = [{
    'C': [0.1, 1, 10, 100, 1000],
    'class_weight': [{1: w} for w in list(range(1, 11))],
  }]
  t.train(name, svc)
  t.train(name, svc, True)
  t.train_cv(name, svc, parameters, False, True)
  t.train_cv(name, svc, parameters, True, True)
  print_line()

  # SVC
  name = 'SVC'
  svc = SVC(random_state=0)
  parameters = [{
    'kernel': ['rbf'],
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],
    'class_weight': [{1: w} for w in list(range(1, 11))],
  },
  {
    'kernel': ['sigmoid'],
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],
    'class_weight': [{1: w} for w in list(range(1, 11))],
  },
  {
    'kernel': ['polynomial'],
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],
    'degree': list(range(2, 7)),
    'class_weight': [{1: w} for w in list(range(1, 11))],
  }]
  t.train(name, svc, True)
  t.train_cv(name, svc, parameters, True)
  print_line()

  # KNN
  name = 'KNN'
  knn = KNeighborsClassifier()
  parameters = [{
    'n_neighbors': list(range(1, 11)),
    'p': list(range(1, 6)),
    'weights': ['uniform', 'distance'],
  }]
  t.train(name, knn, True)
  t.train_cv(name, knn, parameters, True)
  print_line()

  # GBDT
  name = 'GBDT'
  gbdt = GradientBoostingClassifier()
  parameters = [{
    'n_estimators': [16, 32, 64, 100, 150, 200],
    'learning_rate': [0.0001, 0.001, 0.01, 0.025, 0.05,  0.1, 0.25, 0.5],
  }]
  t.train(name, gbdt)
  t.train(name, gbdt, True)
  t.train_cv(name, gbdt, parameters, False)
  t.train_cv(name, gbdt, parameters, True, True)
  print_line()

print('Finished training')

#joblib.dump(best_clf, 'bard-classifier.pkl')
