from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import ml.sample_generators as s
import ml.train_utils as t
import bardo.utils.db_utils as db
from datetime import datetime
from copy import deepcopy
import numpy as np


### Experimentation configs
TEST_SIZE = 0.25
K = 5
# Train:test datasets
ROOT = 'data/datasets'
SUFFIX = 'savg_norm2'
datasets = [
  (['lsgaleana@gmail.com'], [
    'lsgaleana@gmail.com',
  ]),
  ([
    'lsgaleana@gmail.com',
    'sheaney@gmail.com',
    'others',
  ], [
    'lsgaleana@gmail.com',
  ]),
]
# Generators sample the data
generators = [
  s.BinaryTestGen(),
  s.BinaryTestGen({4, 5, 7}),
  s.BinaryTestGen(neg_train={1, 2, 6}),
  s.BinaryTestGen({4, 5, 7}, {1, 2, 6}),

  s.VeryBinaryTestGen(),
  s.VeryBinaryTestGen(very=1),
  s.VeryBinaryTestGen(very=-1),

  s.VeryBinaryTestGen({4, 5, 7}),
  s.VeryBinaryTestGen({4, 5, 7}, very=1),
  s.VeryBinaryTestGen({4, 5, 7}, very=-1),

  s.VeryBinaryTestGen(neg_train={1, 2, 6}),
  s.VeryBinaryTestGen(neg_train={1, 2, 6}, very=1),
  s.VeryBinaryTestGen(neg_train={1, 2, 6}, very=-1),

  s.VeryBinaryTestGen({4, 5, 7}, {1, 2, 6}),
  s.VeryBinaryTestGen({4, 5, 7}, {1, 2, 6}, very=1),
  s.VeryBinaryTestGen({4, 5, 7}, {1, 2, 6}, very=-1),
#
#  s.TernaryTestGen(),
#  s.TernaryTestGen(neu_train={3, 6}),
#  s.TernaryTestGen(neu_train={3, 7}),
#  s.TernaryTestGen(neu_train={3, 6, 7}),
#
#  s.TernaryTestGen({4, 5, 7}),
#  s.TernaryTestGen(neg_train={1, 2, 6}),
#  s.TernaryTestGen({4, 5, 7}, {1, 2, 6}),
#
#  s.TernaryTestGen({4, 5, 7}, neu_train={3, 6}),
#  s.TernaryTestGen(neg_train={1, 2, 6}, neu_train={3, 7}),
]
# CV parameters
sp = [{
  'model__gamma': ['scale', 'auto', 0.002, 0.0025, 0.003, 0.0035, 0.004],
}]
# Models
exp_configs = [
  {
    'name': 'Linear SVC',
    'model': LinearSVC(dual=False), 
    'modes': [
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
# Experiment configs
# One per training dataset, generator, config and config mode
metrics = {}
# Train users map to datasets
for train_users, test_users in datasets:
  t.print_line()

  train_files = [db.escape_bardo_id(user) for user in train_users]
  train_datasets = [f'{ROOT}/{file_}_{SUFFIX}.txt' for file_ in train_files]

  # Models contains a map from generator -> (model_name, (model, cv))
  models = {}
  for generator in generators:
    generator_ = deepcopy(generator)
    print(generator_.get_name())
    t.print_line()
  
    models[generator_] = {}
    for config in exp_configs:
      for mode in config['modes']:
        model_name = f'{config["name"]}:{"+".join(train_users)}'
        tu = t.TrainUtil(
          name=model_name,
          model=deepcopy(config['model']),
          datasets=[deepcopy(generator).set(d) for d in train_datasets],
          test_size=TEST_SIZE,
          standardize=mode['standardize'],
          k=K,
          params=mode['params'],
        )

        # We store an instance of model (for later) and its CV metrics
        models[generator_][model_name] = (tu.train(),
          tu.get_cv_metrics() if 'CV' in test_users else None)
    t.print_line()

  test_files = [db.escape_bardo_id(user) for user in test_users]
  test_datasets = [f'{ROOT}/{file_}_{SUFFIX}.txt' for file_ in test_files]

  print('Generating metrics')
  # Test users map to test datasets
  for test_user, test_dataset in zip(test_users, test_datasets):
    if test_user not in metrics:
      metrics[test_user] = {}

    for generator, tus in models.items():
      if generator.get_name() not in metrics[test_user]:
        metrics[test_user][generator.get_name()] = {}

      for model_name, tu in tus.items():
        if test_user == 'CV':
          metrics[test_user][generator.get_name()][model_name] = tu[1]
        else:
          # We need the model instances to generate dataset-specific metrics
          metrics[test_user][generator.get_name()][model_name] = \
            tu[0].get_test_metrics(deepcopy(generator).set(test_dataset))

print('Writing metrics')
t.print_line()
now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
with open(f'data/reports/{now}.txt', 'w+') as log_file:
  log_file.write(',,Train Acc,TestAcc,Test 1 Pr,Test 1 Rec,Test 0 Pr,Test 0 Rec\n')
  for test_user, models in metrics.items():
    log_file.write(f'{test_user}\n')
    for generator, tm in models.items():
      log_file.write(f'{generator}')
      for model_name, m in tm.items():
        log_file.write(f',{model_name},{m["train_acc"]},{m["test_acc"]},{m["test_pr_1"]},{m["test_rec_1"]},{m["test_pr_0"]},{m["test_rec_0"]}\n')

  log_file.write(f'\n{TEST_SIZE} {SUFFIX}')
print('Finished training')
