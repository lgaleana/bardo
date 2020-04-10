from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import train_utils as t
import sample_generators as s
import spotify_utils as su
import random
from time import time


SECS_IN_10_MIN = 600

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
DATASET = 'datasets/dataset_all.txt'
TEST_SIZE = 0
train_configs = [
#  {
#    'name': 'svc_cv_very_balanced',
#    'model': SVC(random_state=0),
#    'generator': s.VeryBinaryTestGen(DATASET, TEST_SIZE, 3, 4, True, True),
#    'standardize': True,
#    'params': svc_params,
#  },
#  {
#    'name': 'gbdt_cv_very',
#    'model': GradientBoostingClassifier(random_state=0),
#    'generator':   s.VeryBinaryTestGen(DATASET, TEST_SIZE, 3, 4),
#    'standardize': True,
#    'params': gbdt_params,
#  },
  {
    'name': 'gbdt_very_high',
    'model': GradientBoostingClassifier(random_state=0),
    'generator': s.VeryBinaryTestGen(DATASET, TEST_SIZE, 3, 4, False, True),
    'standardize': True,
    'params': False,
  },
#  {
#    'name': 'svc_cv_very',
#    'model': SVC(random_state=0),
#    'generator':   s.VeryBinaryTestGen(DATASET, TEST_SIZE, 3, 4),
#    'standardize': True,
#    'params': svc_params,
#  },
]


### Train production classifiers
# And load labeled tracks so we don't recommend them
classifiers = {}
tracks = []
def load_prod_classifiers():
  print('---Loading tracks DB---')
  with open('datasets/tracks.txt') as f:
    for line in f:
      track = line.strip().split('\t')[1]
      tracks.append(track)
  t.print_line()

  for config in train_configs:
    data = config['generator'].gen()
    tu = t.TrainUtil(
      name=config['name'],
      model=config['model'],
      data=data,
      standardize=config['standardize'],
      params=config['params']
    )
    tu.train()
    classifiers[config['name']] = tu
  print('Finished training')

### Get a playlist with recommendations
# Creates a playlist with {limit} tracks from different classifiers
def generate_recommendations(token, genres, exp_config, limit, plst_name):
  final_playlist = []
  # We will store tracks recommended by every classifier up to a limit
  INDIVIDUAL_LIMIT = 10
  playlists = {}
  for name in classifiers:
    if name in exp_config:
      playlists[name] = {
        'ids': [],
        'names': [],
      }
  if 'random' in exp_config:
    playlists['random'] = {
      'ids': [],
      'names': [],
    }
  
  go_on = True
  start_time = time()
  while go_on and time() - start_time < SECS_IN_10_MIN:
    # We get 100 recommendations
    t.print_line()
    recommendations = su.get_recommendations(token, genres)
    t.print_line()
    for recommendation in recommendations:
      go_on = False
      # Check if track is labeled or has been seen
      if recommendation['name'] not in tracks:
        tracks.append(recommendation['name'])
        features = su.get_tracks_features(token, [recommendation])[0]
        analysis = su.get_track_analysis(token, recommendation)
        # Get predictions from all classifiers
        for name, clf in classifiers.items():
          prediction = clf.predict_prod(features + analysis)
          print(f'  {name} prediction: {prediction}')
          if prediction == 1 and recommendation['name'] not in playlists[name]['names'] and len(playlists[name]['ids']) < INDIVIDUAL_LIMIT:
            playlists[name]['ids'].append(recommendation['id'])
            playlists[name]['names'].append(recommendation['name'])
          print(f'  size: {len(playlists[name]["ids"])}')

        if 'random' in playlists and recommendation['name'] not in playlists['random']['names'] and len(playlists['random']['ids']) < INDIVIDUAL_LIMIT:
          print(f'  random prediction: 1.0')
          playlists['random']['ids'].append(recommendation['id'])
          playlists['random']['names'].append(recommendation['name'])
          print(f'  size: {len(playlists["random"]["ids"])}')
      else:
        print(f'{recommendation["name"]} already labeled')

      for plst in playlists.values():
        if len(plst['ids']) < INDIVIDUAL_LIMIT:
          go_on = True
          break
      if not go_on:
        break
    print(f'{(time() - start_time) / 60} mins elapsed')

  # Save classifier playlists for analysis
  # and put together final playlist
  for name, plst in playlists.items():
    f = open(
      f'datasets/lsgaleana-gmail_com/playlists/{plst_name}_{name}.txt', 'w+',
    )
    for i, track in enumerate(plst['ids']):
      final_playlist.append(track)
      f.write(f'{track}\t{plst["names"][i]}\n')
    f.close()

  random.shuffle(final_playlist)
  return final_playlist
