from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import sample_generators as s
import train_utils as t
import datetime


LOG_TO_FILE = True

### Experimentation configs
# Generators generate different training samples
# We want to test many
DATASET = 'datasets/dataset_all.txt'
TEST_SIZE = 0
generators = [
  s.BinaryTestGen(DATASET, TEST_SIZE, 3, 4),
  s.VeryBinaryTestGen(DATASET, TEST_SIZE, 3, 4),
  s.BinaryTestGen(DATASET, TEST_SIZE, 3, 4, True, False),
  s.BinaryTestGen(DATASET, TEST_SIZE, 3, 4, False, True),
  s.BinaryTestGen(DATASET, TEST_SIZE, 3, 4, True, True),
]

# CV parameters
lsp = [{
  'C': [0.1, 1, 10, 100, 1000],
  'class_weight': [{1: w} for w in list(range(1, 11))],
}]
sp = [{
  'kernel': ['rbf'],
  'C': [0.1, 1, 10, 100, 1000],
  'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],
  'class_weight': [{1: w} for w in list(range(1, 11))],
}]
kp = [{
  'n_neighbors': list(range(1, 11)),
  'p': list(range(1, 6)),
  'weights': ['uniform', 'distance'],
}]
gp = [{
  'n_estimators': [16, 32, 64, 100, 150, 200],
  'learning_rate': [0.0001, 0.001, 0.01, 0.025, 0.05,  0.1, 0.25, 0.5],
}]

# Define what to experiment with
exp_configs = [
  {
    'name': 'Linear SVC',
    'model': LinearSVC(dual=False), 
    'modes': [
      {'standardize': False, 'params': False},
      {'standardize': True, 'params': False},
      {'standardize': False, 'params': lsp},
      {'standardize': True, 'params': lsp},
    ],
  },
  {
    'name': 'SVC',
    'model': SVC(random_state=0), 
    'modes': [
      {'standardize': True, 'params': False},
      {'standardize': True, 'params': sp},
    ],
  },
  {
    'name': 'KNN',
    'model': KNeighborsClassifier(),
    'modes': [
      {'standardize': True, 'params': False},
      {'standardize': True, 'params': kp},
    ],
  },
  {
    'name': 'GBDT',
    'model': GradientBoostingClassifier(random_state=0), 
    'modes': [
      {'standardize': True, 'params': False},
      {'standardize': True, 'params': gp},
    ],
  },
]


### Run experiments
# Whether to log the data to a file
now = str(datetime.datetime.now()).replace(':', '-').replace('.', '_').replace(' ', '_')
log_file = None
if LOG_TO_FILE:
  log_file = open(
    f'reports/{now}.txt',
    'a+',
  )
  log_file.write(',Train Acc,TestAcc,,Train Pr,Test Pr,Test Rec\n')

for generator in generators:
  data = generator.gen()

  if LOG_TO_FILE:
    log_file.write(f'{data.__class__.__name__}\n') 

  # Training of all configs
  for config in exp_configs:
    for mode in config['modes']:
      tu = t.TrainUtil(
        name=config['name'],
        model=config['model'],
        data=data,
        standardize=mode['standardize'],
        params=mode['params'],
      )
      #tu.train()

      train_acc, test_acc, train_pr, test_pr, test_rec = tu.get_cv_metrics()
      if LOG_TO_FILE:
        print('Writting metrics')
        log_file.write(f'{tu.get_name()},{train_acc},{test_acc},,{train_pr},{test_pr},{test_rec}\n')
      else:
        print(f'Train Acc Test Acc | Train Pr Test Pr Test Rec')
        print(f'   {train_acc:.2f}      {test_acc:.2f}      {train_pr:.2f}     {test_pr:.2f}    {test_rec:.2f}')

  t.print_line()
if LOG_TO_FILE:
  log_file.close()
print('Finished training')
