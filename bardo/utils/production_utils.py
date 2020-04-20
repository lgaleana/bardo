from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import ml.train_utils as t
import ml.sample_generators as s
import bardo.utils.spotify_utils as su
from random import shuffle
from time import time
from copy import deepcopy

### Train configs
# Classifiers initial configs
svc = SVC(random_state=0)
linear_svc = LinearSVC(dual=False)
knn = KNeighborsClassifier()
gbdt = GradientBoostingClassifier(random_state=0)
# Configs
# These classifiers were picked through experimentation
DATASET = 'datasets/dataset_all.txt'
K = 5
train_configs = [
  {
    'name': 'svc_bottom_high',
    'model': SVC(C=0.95, random_state=0),
    'generator': s.VeryBinaryTestGen(DATASET, 3, 4, False, True, -1),
    'standardize': True,
  },
  {
    'name': 'svc_very',
    'model': SVC(random_state=0),
    'generator': s.VeryBinaryTestGen(DATASET, 3, 4),
    'standardize': True,
  },
  {
    'name': 'gbdt_bottom_high',
    'model': GradientBoostingClassifier(learning_rate=0.095, random_state=0),
    'generator': s.VeryBinaryTestGen(DATASET, 3, 4, False, True, -1),
    'standardize': True,
  },
]


### Train production classifiers
classifiers = {}
def load_prod_classifiers():
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
def gen_recs(token, genres, exp_config,  market, profile, slimit, tlimit):
  # Use profile as seed
  pos_tracks = list(map(
    lambda track: track['id'],
    filter(lambda track: track['stars'] >= 4, profile),
  ))
  shuffle(pos_tracks)
  profile = list(map(lambda track: track['name'], profile))

  # We want tracks from every classifier
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
  seeds = {'genres': genres}

  go_on = True
  start_time = time()
  use_random = False
  while go_on and time() - start_time < tlimit:
    # Add seed tracks
    if not use_random and len(pos_tracks) > 0:
      seeds['tracks'] = [pos_tracks.pop(0)]
      rlimit = 10
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
      if recommendation['is_playable'] and recommendation['name'] not in profile:
        nlabel += 1
        profile.append(recommendation['name'])
        features = su.get_tracks_features(token, [recommendation])[0]
        analysis = su.get_track_analysis(token, recommendation)
        # Get predictions from all classifiers
        for name, clf in classifiers.items():
          if name in exp_config:
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
    use_random = not use_random
    t.print_line()
    print(f'{(time() - start_time) / 60} mins elapsed')
    print(f'{nlabel} new track labeled')

  # Add final playlist
  final_playlist = {
    'ids': [],
    'names': [],
  }
  plst_copy = deepcopy(playlists)
  go_on = True
  while len(final_playlist['ids']) < slimit and go_on:
    go_on = False
    for plst in plst_copy.values():
      if len(plst['ids']) > 0:
        go_on = True
        track = plst['ids'].pop(0)
        name = plst['names'].pop(0)
        if track not in final_playlist['ids']:
          final_playlist['ids'].append(track)
          final_playlist['names'].append(name)
  cont = list(zip(final_playlist['ids'], final_playlist['names']))
  shuffle(cont)
  final_playlist['ids'], final_playlist['names'] = zip(*cont)

  return playlists, final_playlist
