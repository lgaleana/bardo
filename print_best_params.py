from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import train_utils as t
import sample_generators as s
from copy import deepcopy

### Fixed params
# Classifiers
svc = SVC(random_state=0)
linear_svc = LinearSVC(dual=False)
knn = KNeighborsClassifier()
gbdt = GradientBoostingClassifier(random_state=0)
# Cross validation
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

### Training configs
DATASET = 'datasets/dataset_all.txt'
TEST_SIZE = 0.25
K = 5
train_configs = [
  {
    'name': 'svc_very',
    'model': deepcopy(svc),
    'generator': s.VeryBinaryTestGen(DATASET, 3, 4),
    'standardize': True,
    'params': sp,
  },
]

for config in train_configs:
  tu = t.TrainUtil(
    name=config['name'],
    model=config['model'],
    data=config['generator'],
    test_size=TEST_SIZE,
    standardize=config['standardize'],
    k=K,
    params=config['params']
  )
  tu.train()
  print(tu.get_params())
