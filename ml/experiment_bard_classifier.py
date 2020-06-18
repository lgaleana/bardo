from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import ml.sample_generators as s
import ml.train_utils as t
from datetime import datetime
from copy import deepcopy
import numpy as np


### Experimentation configs
# Generators generate different training samples
# We want to test many
TEST_SIZE = 0.25
K = 5
ROOT = 'data/datasets'
datasets = [
  f'{ROOT}/lsgaleana-gmail_com_test.txt',
  f'{ROOT}/sheaney-gmail_com_test.txt',
  f'{ROOT}/others_test.txt',
]
generators = [
  s.BinaryTestGen(neg_train={1, 2, 6}),
  s.BinaryTestGen({3, 4, 5}, {1, 2, 6}),
  s.BinaryTestGen(neg_train={1, 2, 3, 6}),
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
    ],
  },
  {
    'name': 'SVC',
    'model': SVC(random_state=0), 
    'modes': [
      {'standardize': True, 'params': None},
    ],
  },
  {
    'name': 'KNN',
    'model': KNeighborsClassifier(),
    'modes': [
      {'standardize': True, 'params': None},
    ],
  },
  {
    'name': 'GBDT',
    'model': GradientBoostingClassifier(random_state=0), 
    'modes': [
      {'standardize': True, 'params': None},
    ],
  },
]


### Run experiments
now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file = open(f'data/reports/{now}.txt', 'w+')
log_file.write(',Train Acc,TestAcc,Test 1 Pr,Test 1 Rec,Test 0 Pr,Test 0 Rec\n')

models = {}
# Create all train configs
# One per generator, config, config mode and dataset
# (plus one with all datasets)
for generator in generators:
  print(generator.get_name())
  t.print_line()

  models[generator] = []
  for config in exp_configs:
    for mode in config['modes']:
      all_generators = []
      for dataset in datasets:
        chunks = dataset.split('/')
        user = chunks[len(chunks) - 1]
        user_generator = generator.set(dataset)

        models[generator].append(t.TrainUtil(
          name=f'{config["name"]}+{user}',
          model=deepcopy(config['model']),
          datasets=[deepcopy(user_generator)],
          test_size=TEST_SIZE,
          standardize=mode['standardize'],
          k=K,
          params=mode['params'],
        ).train())
        all_generators.append(deepcopy(user_generator))

      models[generator].append(t.TrainUtil(
        name=f'{config["name"]}+ALL',
        model=deepcopy(config['model']),
        datasets=all_generators,
        test_size=TEST_SIZE,
        standardize=mode['standardize'],
        k=K,
        params=mode['params'],
      ).train())
  t.print_line()

# Add a fictitional dataset, which is cross-validation, for metrics purpose
datasets.insert(0, 'CV')
# Experiment with train configs
print('Generating metrics')
t.print_line()
for dataset in datasets:
  if dataset == 'CV':
    print(f'CV')
    log_file.write('CV\n')
  else:
    chunks = dataset.split('/')
    user = chunks[len(chunks) - 1]
    print(f'{user}')
    log_file.write(f'{user}\n')
  t.print_line()

  for generator, tus in models.items():
    print(f'{generator.get_name()}')
    log_file.write(f'{generator.get_name()}\n')
    for tu in tus:
      if dataset == 'CV':
        cv = tu.get_cv_metrics()
        log_file.write(f'{tu.get_name()},{cv["train_acc"]},{cv["test_acc"]},{cv["test_pr_1"]},{cv["test_rec_1"]},{cv["test_pr_0"]},{cv["test_rec_0"]}\n')
      else:
        tm = tu.get_test_metrics(deepcopy(generator).set(dataset))
        log_file.write(f'{tu.get_name()},{tm["train_acc"]},{tm["test_acc"]},{tm["test_pr_1"]},{tm["test_rec_1"]},{tm["test_pr_0"]},{tm["test_rec_0"]}\n')
  log_file.write('\n')
  t.print_line()
log_file.close()
print('Finished training')
