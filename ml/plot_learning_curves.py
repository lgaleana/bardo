from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import ml.sample_generators as s
import ml.train_utils as t

### Learn configs
# Curve params
PLOT = False
SCORER = 'f1'
POINTS = 20
# Configs
DATASET = 'datasets/dataset_test2.txt'
K = 5
TEST_SIZE = 0.25
learn_configs = [
  {
    'name': 'svc_mixed',
    'model': SVC(C=1.22, random_state=0),
    'generator': s.BinaryTestGen('datasets/dataset_test.txt'),
    'standardize': True,
  },
  {
    'name': 'svc_top_mixed',
    'model': SVC(C=1.58, random_state=0),
    'generator': s.VeryBinaryTestGen('datasets/dataset_test.txt'),
    'standardize': True,
  },
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
  print(f'Train Acc Test Acc | Test Pr1 Test Rec1 Test Pr0 Test Rec0')
  print(f'  {tm["train_acc"]:.2f}      {tm["test_acc"]:.2f}       {tm["test_pr_1"]:.2f}      {tm["test_rec_1"]:.2f}      {tm["test_pr_0"]:.2f}      {tm["test_rec_0"]:.2f}')
  if PLOT:
    tu.plot_learning_curve(SCORER, POINTS)
t.print_line()
print('Finished plotting')
