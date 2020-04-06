from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import train_utils as t
import sample_generators as s
import spotify_utils as su
import random


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
DATASET = 'datasets/dataset_all.txt'
train_configs = [
  {
    'name': 'gbdt_very',
    'model': GradientBoostingClassifier(random_state=0),
    'generator': s.VeryBinaryTestGen(DATASET, 0, 3, 4),
    'standardize': True,
    'params': False,
  },
  {
    'name': 'gbdt_cv_very',
    'model': GradientBoostingClassifier(random_state=0),
    'generator': s.VeryBinaryTestGen(DATASET, 0, 3, 4),
    'standardize': True,
    'params': gbdt_params,
  },
  {
    'name': 'gbdt',
    'model': GradientBoostingClassifier(random_state=0),
    'generator': s.BinaryTestGen(DATASET, 0, 3, 4, True, True),
    'standardize': True,
    'params': False,
  },
  {
    'name': 'gbdt_cv',
    'model': GradientBoostingClassifier(random_state=0),
    'generator': s.BinaryTestGen(DATASET, 0, 3, 4, True, True),
    'standardize': True,
    'params': gbdt_params,
  },
]

### Load labeled tracks
# So that we don't recommend them
print('---Loading tracks DB---')
tracks = []
with open('datasets/tracks.txt') as f:
  for line in f:
    track = line.strip().split('\t')[1]
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
      params=config['params']
    )
    tu.train()
    classifiers[config['name']] = tu
  print('Finished training')

### Get a playlist with recommendations
# Creates a playlist with {limit} tracks from different classifiers
def generate_recommendations(token, genres, limit, plst_name):
  final_playlist = []
  # We will store tracks recommended by every classifier up to a limit
  INDIVIDUAL_LIMIT = 10
  playlists = {}
  for name in classifiers:
    playlists[name] = {
      'ids': [],
      'names': [],
    }
  all_recs = set()

#  # Special playlists
#  playlists['random'] = []
#  RANDOM_LIMIT = 5
  
  go_on = True
  while go_on:
    # We get 100 recommendations
    t.print_line()
    recommendations = su.get_recommendations(token, genres)
    t.print_line()
    for recommendation in recommendations:
      go_on = False
      # Check if track is labeled or has been seen
      if recommendation['name'] not in tracks and recommendation['name'] not in all_recs:
        features = su.get_tracks_features(token, [recommendation])[0]
        analysis = su.get_track_analysis(token, recommendation)
        # Get predictions from all classifiers
        for name, clf in classifiers.items():
          prediction = clf.predict_prod(features + analysis)
          print(f'  {name} prediction: {prediction}')
          if prediction == 1 and recommendation['name'] not in playlists[name]['names'] and len(playlists[name]) < INDIVIDUAL_LIMIT:
            playlists[name]['ids'].append(recommendation['id'])
            playlists[name]['names'].append(recommendation['name'])
            all_recs.add(recommendations['id'])
          print(f'  size: {len(playlists[name])}')
      else:
        print(f'{recommendation["name"]} already labeled')

#      # Special playlists
#      if len(playlists['random']) < RANDOM_LIMIT:
#        print(f'  random prediction: 1')
#        playlists['random'].append(recommendation['name'])
#        print(f'  size: {len(playlists["random"])}')
#        if recommendations['id'] not in final_playlist:
#          final_playlist.append(recommendation['id'])
  
      for plst in playlists.values():
        if len(plst['ids']) < INDIVIDUAL_LIMIT:
          go_on = True
          break
      if not go_on:
        break

  # Save classifier playlists for analysis
  for name, plst in playlists.items():
    f = open(f'playlists/{plst_name}_{name}.txt', 'w')
    for track in plst['names']:
      f.write(f'{track}\n')
    f.close()

  # Put together final playlist
  for _ in range(INDIVIDUAL_LIMIT):
    for name, plst in playlists.items():
      if plst['ids'][0] not in final_playlist:
        final_playlist.append(plst['ids'][0])
      playlists[name]['ids'] = plst['ids'][1:]

  random.shuffle(final_playlist)
  return final_playlist

### Utils
# Can insert if playlist is not larger than any other by more than 1
def can_insert(playlists, name):
  min_cnt = min(map(lambda plst: len(plst), playlists.values()))
  return len(playlists[name]) - min_cnt == 0
