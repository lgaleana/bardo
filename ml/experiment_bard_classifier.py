from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import ml.sample_generators as s
import ml.train_utils as t
from datetime import datetime
from copy import deepcopy


LOG_TO_FILE = True
DO_TEST = True

### Experimentation configs
# Generators generate different training samples
# We want to test many
DATASET = 'datasets/dataset_all.txt'
TEST_SIZE = 0.25
K = 5
generators = [
  s.BinaryTestGen(DATASET, 3, 4),
  s.VeryBinaryTestGen(DATASET, 3, 4),
  s.BinaryTestGen(DATASET, 3, 4, False, True),
  s.VeryBinaryTestGen(DATASET, 3, 4, False, True, -1),
  s.VeryBinaryTestGen(DATASET, 3, 4, False, True),
  s.BinaryTestGen(DATASET, 3, 4, True, True),
  s.VeryBinaryTestGen(DATASET, 3, 4, True, True, 1),
  s.VeryBinaryTestGen(DATASET, 3, 4, True, True),
]

# CV parameters
lsp = [{
  'C': [0.1, 1, 10, 100, 1000],
  'class_weight': [{1: w} for w in list(range(1, 11))],
}]
sp = [{
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
      {'standardize': True, 'params': None},
      {'standardize': True, 'params': lsp},
    ],
  },
  {
    'name': 'SVC',
    'model': SVC(random_state=0), 
    'modes': [
      {'standardize': True, 'params': None},
      {'standardize': True, 'params': sp},
    ],
  },
  {
    'name': 'KNN',
    'model': KNeighborsClassifier(),
    'modes': [
    ],
  },
  {
    'name': 'GBDT',
    'model': GradientBoostingClassifier(random_state=0), 
    'modes': [
      {'standardize': True, 'params': None},
      {'standardize': True, 'params': gp},
    ],
  },
]


### Run experiments
# Whether to log the data to a file
now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
log_file = None
if LOG_TO_FILE:
  log_file = open(f'datasets/reports/{now}.txt', 'w+')
  header = ',Train Acc,TestAcc,,Test Pr 1,Test Rec 1'
  if DO_TEST:
    header += ',T:,TestAcc,,Test Pr 1,Test Rec 1,,Test Pr 0,Test Rec 0\n'
  log_file.write(header)

for generator in generators:
  print(generator.get_name())
  t.print_line()

  if LOG_TO_FILE:
    log_file.write(f'{generator.get_name()}\n') 

  # Training of all configs
  for config in exp_configs:
    for mode in config['modes']:
      tu = t.TrainUtil(
        name=config['name'],
        model=deepcopy(config['model']),
        data=deepcopy(generator),
        test_size=TEST_SIZE,
        standardize=mode['standardize'],
        k=K,
        params=mode['params'],
      )

      cv = tu.get_cv_metrics()
      if DO_TEST:
        tu.train()
        tm = tu.get_test_metrics()

      if LOG_TO_FILE:
        print('---Writting metrics---')
        metrics = f'{tu.get_name()},{cv["train_acc"]},{cv["test_acc"]},,{cv["test_pr_1"]},{cv["test_rec_1"]}'
        if DO_TEST:
          metrics += f',,{tm["test_acc"]},,{tm["test_pr_1"]},{tm["test_rec_1"]},,{tm["test_pr_0"]},{tm["test_rec_0"]}\n'
        log_file.write(metrics)
      else:
        print(f'Train Acc Test Acc | Train Pr Test Pr Test Rec')
        print(f'   {train_acc:.2f}      {test_acc:.2f}      {train_pr:.2f}     {test_pr:.2f}    {test_rec:.2f}')
        if DO_TEST:
          print(f'TEST-')
          print(f'           {test_acc_t:.2f}           {test_pr_t:.2f}  {test_rec_t:.2f}')

  t.print_line()
if LOG_TO_FILE:
  log_file.close()
print('Finished training')
