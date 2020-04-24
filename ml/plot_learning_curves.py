from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import ml.sample_generators as s
import ml.train_utils as t

### Learn configs
# Curve params
PLOT = True
SCORER = 'f1'
POINTS = 20
# Configs
DATASET = 'datasets/dataset_all.txt'
K = 5
TEST_SIZE = 0.25
learn_configs = [
  {
    'name': 'svc_high',
    'model': SVC(random_state=0),
    'generator': s.BinaryTestGen(DATASET, 3, 4, False, True),
    'standardize': True,
  },
#  {
#    'name': 'svc_very_high',
#    'model': SVC(random_state=0),
#    'generator': s.VeryBinaryTestGen(DATASET, 3, 4, False, True),
#    'standardize': True,
#  },
#  {
#    'name': 'svc_very_balanced',
#    'model': SVC(random_state=0),
#    'generator': s.BinaryTestGen(DATASET, 3, 4, True, True),
#    'standardize': True,
#  },
]


### Plot curves
# Training of all configs
for config in learn_configs:
  tu = t.TrainUtil(
    name=config['name'],
    model=config['model'],
    data=config['generator'],
    test_size=TEST_SIZE,
    standardize=config['standardize'],
    k=K,
  )
  tu.train()
  tm = tu.get_test_metrics()
  print(f'{tm["test_acc"]} {tm["test_pr_1"]} {tm["test_rec_1"]} {tm["test_pr_0"]} {tm["test_rec_0"]}')
  tu.plot_learning_curve(SCORER, POINTS)
t.print_line()
print('Finished plotting')
