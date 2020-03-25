from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import train_utils as t
import sample_generators as s
import spotify_utils as su


### Fixed params
# Classifiers
linear_svc = LinearSVC(dual=False)
svc = SVC(random_state=0)
knn = KNeighborsClassifier()
gbdt = GradientBoostingClassifier(random_state=0)
# Cross validation
linear_svc_params = [{
  'C': [0.1, 1, 10, 100, 1000],
  'class_weight': [{1: w} for w in list(range(1, 11))],
}]
svc_params = [{
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
}]
knn_params = [{
  'n_neighbors': list(range(1, 11)),
  'p': list(range(1, 6)),
  'weights': ['uniform', 'distance'],
}]
gbdt_params = [{
  'n_estimators': [16, 32, 64, 100, 150, 200],
  'learning_rate': [0.0001, 0.001, 0.01, 0.025, 0.05,  0.1, 0.25, 0.5],
}]

### Training configs
# These classifiers were picked through experimentation
DATASET = 'dataset.txt'
TEST_SIZE = 0.25
CV = 6
train_configs = [
  {
    'name': 'svc_pos_neg',
    'model': svc,
    'generator': s.PosAndNegGen(DATASET, TEST_SIZE),
    'standardize': True,
    'cv': False,
    'parameters': None,
  },
  {
    'name': 'svc_very_pos_neg',
    'model': svc,
    'generator': s.VeryPosAndNegGen(DATASET, TEST_SIZE),
    'standardize': True,
    'cv': False,
    'parameters': None,
  },
  {
    'name': 'knn_pos_neg_neutral_train',
    'model': knn,
    'generator': s.PosNegAndNeutralTrainGen(DATASET, TEST_SIZE),
    'standardize': True,
    'cv': CV,
    'parameters': knn_params,
  },
  {
    'name': 'gbdt_pos_neutral_neg',
    'model': gbdt,
    'generator': s.PosAndNeutralNegGen(DATASET, TEST_SIZE),
    'standardize': True,
    'cv': CV,
    'parameters': gbdt_params,
  },
]

### Load labeled tracks
# So that we don't recommend them
print('---Loading tracks DB---')
tracks = []
with open('./tracks.txt') as f:
  for line in f:
    track = line.strip()
    tracks.append(track)
t.print_line()

### Train production classifiers
classifiers = {}
def load_prod_classifiers():
  for config in train_configs:
    data = config['generator'].gen()
    tu = t.TrainUtil(
      name=config['name'],
      model=config['model'],
      data=data,
      standardize=config['standardize'],
      cv=config['cv'],
      parameters=config['parameters']
    )
    tu.train()
    classifiers[config['name']] = tu
  print('Finished training')

# The general idea is to make a playlist with {limit} tracks
# per classifier. On top, two other lists will include the
# average of the classifiers and one without classifiers
def generate_recommendations(token, genres, limit):
  playlists = {}
  for name in classifiers:
    playlists[name] = []
  playlists['random'] = []
#  playlists['avg'] = []
  
  go_on = True
  while go_on:
    # We get 100 recommendations
    t.print_line()
    recommendations = su.get_recommendations(token, genres)
    t.print_line()
    for recommendation in recommendations:
      go_on = False
      if recommendation['name'] not in tracks:
        # Track is not labeled
        features = su.get_track_features(token, recommendation)
        # Get predictions from all classifiers
        predictions_sum = 0.0
        for name, clf in classifiers.items():
          # Standardize features
          prediction = clf.predict_prod(features)
          predictions_sum += prediction
          print(f'  {name} prediction: {prediction}')
          # We limit the number of tracks in the playlist
          # And we only care about positives
          if len(playlists[name]) < limit and recommendation['id'] not in playlists[name] and prediction == 1:
            playlists[name].append(recommendation['id'])
      
        # Special cases
        print(f'  random prediction: 1')
        if len(playlists['random']) < limit and recommendation['id'] not in playlists['random']:
          playlists['random'].append(recommendation['id'])
        #prediction = predictions_sum / (len(classifiers) - 1)
        #print(f'  avg prediction: {prediction}')
        #if len(playlists['avg']) < limit and recommendation['id'] not in playlists['avg'] and prediction >= 0.5:
        #  playlists['avg'].append(recommendation['id'])
      else:
        print(f'{recommendation["name"]} already labeled')
  
      # Check if playlists are full
      for name, plst in playlists.items():
        if len(plst) < limit:
          go_on = True
          break
      if not go_on:
        break

  return playlists
