from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import ml.sample_generators as s
import ml.train_utils as t
from datetime import datetime
from copy import deepcopy
import numpy as np


LOG_TO_FILE = True
DO_TEST = True

### Experimentation configs
# Generators generate different training samples
# We want to test many
TEST_SIZE = 0.25
K = 5
ROOT = 'data/datasets'
datasets = [f'{ROOT}/lsgaleana-gmail_com_test.txt', f'{ROOT}/sheaney-gmail_com_test.txt']
generators = [
  s.BinaryTestGen(),
  s.BinaryTestGen(pos_train={4, 5, 7}),
  s.BinaryTestGen(neg_train={1, 2, 6}),
  s.BinaryTestGen(pos_train={4, 5, 7}, neg_train={1, 2, 6}),
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
# Whether to log the data to a file
now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file = None
if LOG_TO_FILE:
  log_file = open(f'data/reports/{now}.txt', 'w+')
  log_file.write(',,Train Acc,TestAcc,Test 1 Pr,Test 1 Rec,Test 0 Pr,Test 0 Rec\n')

for generator in generators:
  print(generator.get_name())
  t.print_line()

  if LOG_TO_FILE:
    log_file.write(f'{generator.get_name()}\n') 

    # Training of all configs
    for config in exp_configs:
      for mode in config['modes']:
        tus = []
        all_generators = []
        for dataset in datasets:
          user_generator = generator.set(dataset)
          tus.append(t.TrainUtil(
            name=config['name'],
            model=deepcopy(config['model']),
            datasets=[deepcopy(user_generator)],
            test_size=TEST_SIZE,
            standardize=mode['standardize'],
            k=K,
            params=mode['params'],
          ))
          all_generators.append(deepcopy(user_generator))
        tus.append(t.TrainUtil(
          name=config['name'],
          model=deepcopy(config['model']),
          datasets=all_generators,
          test_size=TEST_SIZE,
          standardize=mode['standardize'],
          k=K,
          params=mode['params'],
        ))

        for tu in tus:
          cv = tu.get_cv_metrics()
          if DO_TEST:
            tu.train()
            test_metrics = {}
            for user_generator in tu.get_datasets():
              dataset = user_generator.get_dataset_name()
              test_metrics[dataset] = tu.get_test_metrics(dataset)

          if LOG_TO_FILE:
            print('---Writting metrics---')
            log_file.write(f'{tu.get_name()}')
            log_file.write(f',CV,{cv["train_acc"]},{cv["test_acc"]},{cv["test_pr_1"]},{cv["test_rec_1"]},{cv["test_pr_0"]},{cv["test_rec_0"]}\n')
            if DO_TEST:
              metrics = []
              for dataset, tm in test_metrics.items():
                chunks = dataset.split('/')
                user = chunks[len(chunks) - 1]
                cm = [tm['train_acc'],tm['test_acc'],tm['test_pr_1'],tm['test_rec_1'],tm['test_pr_0'],tm['test_rec_0']]
                mstr = ','.join([str(m) for m in cm])
                log_file.write(f',{user},{mstr}\n')
                metrics.append(cm)
              mstr = ','.join([str(m) for m in np.mean(metrics, axis=0)])
              #log_file.write(f',AVG,{mstr}\n')

  t.print_line()
if LOG_TO_FILE:
  log_file.close()
print('Finished training')
