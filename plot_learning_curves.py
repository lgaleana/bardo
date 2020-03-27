from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import sample_generators as s
import train_utils as t
import time


### Learning configs
DATASET = 'dataset.txt'
TEST_SIZE = 0.25
CV = 6
learn_configs = [
  {
    'name': 'Linear SVC',
    'model': LinearSVC(dual=False), 
    'modes': [
    ],
    'parameters': [{
      'C': [0.1, 1, 10, 100, 1000],
      'class_weight': [{1: w} for w in list(range(1, 11))],
    }],
  },
  {
    'name': 'SVC',
    'model': SVC(random_state=0), 
    'modes': [
    ],
    'parameters': [{
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
    }],
  },
  {
    'name': 'KNN',
    'model': KNeighborsClassifier(),
    'modes': [
      {
        'generator': s.PosAndNeutralNegGen(DATASET, TEST_SIZE),
        'standardize': True, 
        'cv': CV,
      },
      {
        'generator': s.VeryPosAndNeutralNegGen(DATASET, TEST_SIZE),
        'standardize': True, 
        'cv': CV,
      },
    ],
    'parameters': [{
      'n_neighbors': list(range(1, 11)),
      'p': list(range(1, 6)),
      'weights': ['uniform', 'distance'],
    }],
  },
  {
    'name': 'GBDT',
    'model': GradientBoostingClassifier(random_state=0), 
    'modes': [
    ],
    'parameters': [{
      'n_estimators': [16, 32, 64, 100, 150, 200],
      'learning_rate': [0.0001, 0.001, 0.01, 0.025, 0.05,  0.1, 0.25, 0.5],
    }],
  },
]


### Plot curves
# Training of all configs
for config in learn_configs:
  for mode in config['modes']:
    data = mode['generator'].gen()
    tu = t.TrainUtil(
      name=config['name'],
      model=config['model'],
      data=data,
      standardize=mode['standardize'],
      cv=mode['cv'],
      parameters=config['parameters']
    )
    tu.train()
    tu.plot_learning_curve(int(1 / TEST_SIZE))
t.print_line()
print('Finished plotting')
