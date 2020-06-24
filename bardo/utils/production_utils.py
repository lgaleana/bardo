from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import ml.train_utils as t
import ml.sample_generators as s
import ml.feature_generator as fg
import bardo.utils.spotify_utils as su
import bardo.utils.db_utils as db
from random import shuffle
from time import time
from copy import deepcopy
import concurrent.futures as f

### Train configs
# Classifiers initial configs
svc = SVC(random_state=0)
linear_svc = LinearSVC(dual=False)
knn = KNeighborsClassifier()
gbdt = GradientBoostingClassifier(random_state=0)
# Configs
# These classifiers were picked through experimentation
K = 5
ROOT = 'data/datasets'
train_configs = [{
  'name': 'svc_sec75_all',
  'model': SVC(random_state=0),
  'datasets': [s.BinaryTestGen().set(f'{ROOT}/lsgaleana-gmail_com_s75.txt'), 
    s.BinaryTestGen().set(f'{ROOT}/sheaney-gmail_com_s75.txt'),
    s.BinaryTestGen().set(f'{ROOT}/others_s75.txt')],
  'standardize': True,
  'holdout': 0.2,
},
{
  'name': 'svc_bottom_4_sec75_all',
  'model': SVC(random_state=0),
  'datasets': [
    s.VeryBinaryTestGen({4, 5, 7}, very=-1).set(
      f'{ROOT}/lsgaleana-gmail_com_s75.txt'), 
    s.VeryBinaryTestGen({4, 5, 7}, very=-1).set(
      f'{ROOT}/sheaney-gmail_com_s75.txt'),
    s.VeryBinaryTestGen({4, 5, 7}, very=-1).set(f'{ROOT}/others_s75.txt')],
  'standardize': True,
  'holdout': 0.2,
},
{
  'name': 'lsvc_very_4_sec75_all',
  'model': LinearSVC(dual=False),
  'datasets': [
    s.VeryBinaryTestGen({4, 5, 7}).set(f'{ROOT}/lsgaleana-gmail_com_s75.txt'), 
    s.VeryBinaryTestGen({4, 5, 7}).set(f'{ROOT}/sheaney-gmail_com_s75.txt'),
    s.VeryBinaryTestGen({4, 5, 7}).set(f'{ROOT}/others_s75.txt')],
  'standardize': True,
  'holdout': 0.25,
},
{
  'name': 'gbdt_sec75_jini',
  'model': GradientBoostingClassifier(random_state=0),
  'datasets': [s.BinaryTestGen().set(f'{ROOT}/sheaney-gmail_com_s75.txt')],
  'standardize': True,
  'holdout': 0.0,
}]
user_configs = {
  'lsgaleana@gmail.com': ['svc_sec75_all', 'svc_bottom_4_sec75_all'],
  'sheaney@gmail.com': ['lsvc_very_4_sec75_all', 'gbdt_sec75_jini'],
}

### Train production classifiers
classifiers = {}
def load_prod_classifiers():
  # Classifiers
  tmp_clfs = {}
  for config in train_configs:
    tu = t.TrainUtil(
      name=config['name'],
      model=config['model'],
      datasets=config['datasets'],
      test_size=config['holdout'],
      standardize=config['standardize'],
      k=K,
    )
    tmp_clfs[config['name']] = tu.train()

  # Assign to users
  for bardo_id, clf_list in user_configs.items():
    classifiers[bardo_id] = {name: tmp_clfs[name] for name in clf_list}

  print('Finished training')

### Get recommendatons from seed tracks or genres
def gen_recs(token, source, genres, history, market, slimit, tlimit, bardo_id):
  users_data = db.load_user_profiles()

  # History seed
  profile = users_data.get(bardo_id, [])
  history_seed = []
  for track in profile:
    if track['stars'] == 4 and 'pos' in history:
      history_seed.append(track['id'])
    elif track['stars'] == 5 and 'very_pos' in history:
      history_seed.append(track['id'])
  shuffle(history_seed)
  profile = list(map(lambda track: track['name'], profile))
  # Genre seed
  seeds = {'genres': genres}

  # We want tracks from every classifier
  clfs = classifiers.get(source, {})
  playlists = {}
  for name in clfs:
    playlists[name] = {
      'ids': [],
      'names': [],
    }
  if source == 'random':
    playlists['random'] = {
      'ids': [],
      'names': [],
    }

  start_time = time()
  go_on = True
  use_random = False
  while go_on and time() - start_time < tlimit:
    go_on = False
    # Add seed tracks
    if not use_random and len(history_seed) > 0:
      seeds['tracks'] = [history_seed.pop(0)]
      rlimit = 10
    else:
      seeds['tracks'] = []
      rlimit = 100

    # We get 100 recommendations, but we might not evaluate all
    recommendations = su.get_recommendations(token, seeds, rlimit, market)
    t.print_line()
    # Concurrent recommendations generation
    with f.ThreadPoolExecutor(max_workers=5) as executor:
      fs = []
      for recommendation in recommendations:
        # Check if track is playable of if it's been labeled
        if recommendation['is_playable'] and recommendation['name'] not in profile:
          # For normal predictions
          if source != 'random':
            fs.append(executor.submit(
              fg.get_audio_and_analysis_features, token, recommendation))
          # Random classifier
          elif recommendation['name'] not in playlists['random']['names'] and len(playlists['random']['ids']) < slimit:
            print(f'  random prediction: 1.0')
            playlists['random']['ids'].append(recommendation['id'])
            playlists['random']['names'].append(recommendation['name'])
            print(f'  size: {len(playlists["random"]["ids"])}')

          profile.append(recommendation['name'])
        else:
          print(f'{recommendation["name"]} already labeled')

      completed, _ = f.wait(fs, 10, f.ALL_COMPLETED)
      for future in completed:
        rec, audio, analysis_ = future.result()
        analysis = analysis_['analysis']
        section = fg.pad_section(analysis_['sections'], 150)
        segment = fg.describe(analysis_['segments'])
        group = fg.get_group_features(bardo_id, rec, users_data)
        user = fg.get_user_features(bardo_id, users_data)

        # Get predictions from all classifiers
        for name, clf in clfs.items():
          prediction = clf.predict_prod(audio + analysis + section + segment + group + user)
          print(f'  {name} prediction: {prediction}')
          if prediction == 1 and rec['name'] not in playlists[name]['names'] and len(playlists[name]['ids']) < slimit:
            playlists[name]['ids'].append(rec['id'])
            playlists[name]['names'].append(rec['name'])
            if rec['id'] not in history_seed:
              history_seed.append(rec['id'])
          print(f'  size: {len(playlists[name]["ids"])}')

      # Toggle use of random as seed
      use_random = not use_random
      t.print_line()
      print(f'{(time() - start_time) / 60} mins elapsed')
      print(f'{len(completed)} new tracks labeled')

      # Should we stop the iteration?
      for plst in playlists.values():
        if len(plst['ids']) < slimit:
          go_on = True

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
