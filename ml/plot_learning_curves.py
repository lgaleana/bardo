from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import sample_generators as s
import train_utils as t

### Learn configs
# Curve params
SCORER = 'f1'
POINTS = 20
# Classifier initial configs
svc = SVC(random_state=0)
linear_svc = LinearSVC(dual=False)
knn = KNeighborsClassifier()
gbdt = GradientBoostingClassifier(random_state=0)
# Configs
DATASET = 'datasets/dataset_all.txt'
K = 5
TEST_SIZE = 0.25
learn_configs = [
  {
    'name': 'svc_very',
    'model': SVC(C=0.995, random_state=0),
    'generator': s.VeryBinaryTestGen(DATASET, 3, 4),
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
  train_acc, test_acc, train_pr, test_pr, test_rec = \
    tu.get_test_metrics()
  print(f'Train Acc Test Acc | Train Pr Test Pr Test Rec')
  print(f'   {train_acc:.2f}      {test_acc:.2f}      {train_pr:.2f}     {test_pr:.2f}    {test_rec:.2f}')
  tu.plot_learning_curve(SCORER, POINTS)
t.print_line()
print('Finished plotting')
