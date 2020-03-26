from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import sample_generators as s
import train_utils as t
import time


LOG_TO_FILE = False
SHOW_LC = True

### Sample generators
# Generators generate different training samples
# We want to test nmany
DATASET = 'dataset.txt'
TEST_SIZE = 0.25
generators = [
  s.PosAndNegGen(DATASET, TEST_SIZE),
  s.VeryPosAndNegGen(DATASET, TEST_SIZE),
  s.PosAndNeutralNegGen(DATASET, TEST_SIZE),
  s.VeryPosAndNeutralNegGen(DATASET, TEST_SIZE),
  s.PosNegAndNeutralTrainGen(DATASET, TEST_SIZE),
  s.VeryPosNegAndNeutralTrainGen(DATASET, TEST_SIZE),
]

### Experimentation configs
# Define what to experiment with
CV = 6
exp_configs = [
  {
    'name': 'Linear SVC',
    'model': LinearSVC(dual=False), 
    'generators': generators,
    'modes': [
      {'standardize': False, 'cv': False},
      {'standardize': True, 'cv': False},
      {'standardize': False, 'cv': CV},
      {'standardize': True, 'cv': CV},
    ],
    'parameters': [{
      'C': [0.1, 1, 10, 100, 1000],
      'class_weight': [{1: w} for w in list(range(1, 11))],
    }],
  },
  {
    'name': 'SVC',
    'model': SVC(random_state=0), 
    'generators': generators,
    'modes': [
      {'standardize': True, 'cv': False},
      {'standardize': True, 'cv': CV},
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
    'generators': generators,
    'modes': [
      {'standardize': True, 'cv': False},
      {'standardize': True, 'cv': CV},
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
    'generators': generators,
    'modes': [
      {'standardize': False, 'cv': False},
      {'standardize': True, 'cv': False},
      {'standardize': False, 'cv': CV},
      {'standardize': True, 'cv': CV},
    ],
    'parameters': [{
      'n_estimators': [16, 32, 64, 100, 150, 200],
      'learning_rate': [0.0001, 0.001, 0.01, 0.025, 0.05,  0.1, 0.25, 0.5],
    }],
  },
]


### Run experiments
tim = str(time.time()).replace('.', '')
for generator in generators:
  data = generator.gen()

  # Whether to log the data to a file
  log_file = None
  if LOG_TO_FILE:
    log_file = open(
      f'reports/{tim}_{data.__class__.__name__}.txt',
      'a+',
    )
    log_file.write(f'{data.__class__.__name__}\n')
    log_file.write(',Train Acc,TestAcc,,Train 1 Pr,Test 1 Pr,,Train 0 Pr,Test 0 Pr\n')

  # Training of all configs
  for config in exp_configs:
    for mode in config['modes']:
      tu = t.TrainUtil(
        name=config['name'],
        model=config['model'],
        data=data,
        standardize=mode['standardize'],
        cv=mode['cv'],
        parameters=config['parameters']
      )
      tu.train()
      tu.print_metrics(log_file)

      if SHOW_LC:
        tu.plot_learning_curve(int(1 / TEST_SIZE))

  if LOG_TO_FILE:
    log_file.close()
  t.print_line()
print('Finished training')
