from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import train_utils as t
import sample_generators as s
import spotify_utils as su
from random import shuffle
from time import time
from copy import deepcopy

# Classifiers
svc = SVC(random_state=0)
linear_svc = LinearSVC(dual=False)
knn = KNeighborsClassifier()
gbdt = GradientBoostingClassifier(random_state=0)
# These classifiers were picked through experimentation
DATASET = 'datasets/dataset_all.txt'
K = 5
train_configs = [
  {
    'name': 'svc_bottom_high',
    'model': deepcopy(svc),
    'generator': s.VeryBinaryTestGen(DATASET, 3, 4, False, True, -1),
    'standardize': True,
  },
  {
    'name': 'svc_very',
    'model': deepcopy(svc),
    'generator': s.VeryBinaryTestGen(DATASET, 3, 4),
    'standardize': True,
  },
  {
    'name': 'svc_cv_very',
    'model': SVC(C=1.0, gamma=0.1, class_weight={1: 2}, random_state=0),
    'generator': s.VeryBinaryTestGen(DATASET, 3, 4),
    'standardize': True,
  },
]


### Train production classifiers
classifiers = {}
tracks = []
pos_tracks = []
def load_prod_classifiers():
  # Load tracks that have already been labeled, so that we don't recommend them
  print('Loading tracks DB')
  pos = []
  with open('datasets/tracks.txt') as f:
    for line in f:
      info = line.strip().split('\t')
      tracks.append(info[1])
      if float(info[len(info) - 1]) == 6:
        pos_tracks.append(info[0])
  shuffle(pos_tracks)

  # Classifiers
  for config in train_configs:
    tu = t.TrainUtil(
      name=config['name'],
      model=config['model'],
      data=config['generator'],
      test_size=0.0,
      standardize=config['standardize'],
      k=K,
    )
    tu.train()
    classifiers[config['name']] = tu
  print('Finished training')

### Get recommendatons from seed tracks or genres
def gen_recs(token, sgenres, exp_config,  market, slimit, tlimit):
  # We want tracks from every classifier
  playlists = {}
  for name in classifiers:
#    if name in exp_config:
    playlists[name] = {
      'ids': [],
      'names': [],
    }
  if 'random' in exp_config:
    playlists['random'] = {
      'ids': [],
      'names': [],
    }
  seeds = {'genres': sgenres}

  go_on = True
  start_time = time()
  nlabel = 0
  while go_on and time() - start_time < tlimit:
    # Add seed tracks
    if nlabel <= 5 and len(pos_tracks) > 0:
      seeds['tracks'] = [pos_tracks.pop(0)]
      rlimit = 20
    else:
      seeds['tracks'] = []
      rlimit = 100

    nlabel = 0
    # We get 100 recommendations
    recommendations = su.get_recommendations(token, seeds, rlimit, market)
    t.print_line()
    for recommendation in recommendations:
      go_on = False
      # Check if track is playable of if it's been labeled
      if recommendation['is_playable'] and recommendation['name'] not in tracks:
        nlabel += 1
        tracks.append(recommendation['name'])
        features = su.get_tracks_features(token, [recommendation])[0]
        analysis = su.get_track_analysis(token, recommendation)
        # Get predictions from all classifiers
        for name, clf in classifiers.items():
          prediction = clf.predict_prod(features + analysis)
          print(f'  {name} prediction: {prediction}')
          if prediction == 1 and recommendation['name'] not in playlists[name]['names'] and len(playlists[name]['ids']) < slimit:
            playlists[name]['ids'].append(recommendation['id'])
            playlists[name]['names'].append(recommendation['name'])
            if recommendation['id'] not in pos_tracks:
              pos_tracks.append(recommendation['id'])
          print(f'  size: {len(playlists[name]["ids"])}')

        if 'random' in playlists and recommendation['name'] not in playlists['random']['names'] and len(playlists['random']['ids']) < slimit:
          print(f'  random prediction: 1.0')
          playlists['random']['ids'].append(recommendation['id'])
          playlists['random']['names'].append(recommendation['name'])
          print(f'  size: {len(playlists["random"]["ids"])}')
      else:
        print(f'{recommendation["name"]} already labeled')

      for plst in playlists.values():
        if len(plst['ids']) < slimit:
          go_on = True
          break
      if not go_on:
        break
    t.print_line()
    print(f'{(time() - start_time) / 60} mins elapsed')
    print(f'{nlabel} new track labeled')

  # Add final playlist
  final_playlist = {
    'ids': [],
    'names': [],
  }
  for name, plst in playlists.items():
    for i, track in enumerate(plst['ids']):
      if track not in final_playlist['ids']:
        final_playlist['ids'].append(track)
        final_playlist['names'].append(plst['names'][i])
  playlists['final'] = final_playlist

  return playlists
