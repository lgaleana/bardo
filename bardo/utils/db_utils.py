import bardo.utils.spotify_utils as su
import os
from datetime import datetime
import json
from collections import OrderedDict

ROOT = 'data'


def load_profile(bardo_id, load_profile=True):
  idn = escape_bardo_id(bardo_id)
  rated_dir = f'{ROOT}/{idn}/feedback'
  profile_dir = f'{ROOT}/{idn}/profile'

  profile_files = []
  if os.path.isdir(rated_dir):
    profile_files = profile_files + list(map(
      lambda filename: f'{rated_dir}/{filename}',
      filter(
        lambda filename: filename.endswith('.txt'),
        os.listdir(rated_dir),
      ),
    ))
  if load_profile and os.path.isdir(profile_dir):
    profile_files = profile_files + list(map(
      lambda filename: f'{profile_dir}/{filename}',
      filter(
        lambda filename: filename.endswith('.txt'),
        os.listdir(profile_dir),
      ),
    ))
  profile_files.sort()

  profile = []
  for path in profile_files:
    for track in load_tracks(path): 
      profile.append(track)

  return profile

def load_profile_deduped(bardo_id):
  profile = {}
  for track in load_profile(bardo_id): 
    if track['stars'] > 0 and track['stars'] <= 5:
      profile[track['id']] = track

  return profile

def load_users_tracks(token):
  users_data = {}
  for bardo_id in load_ids():
    tracks = list(load_profile_deduped(bardo_id).values())

    sp_tracks = []
    offset = 0
    while offset < len(tracks):
      tracks_set = tracks[offset:offset + 50]
      profile_tracks = su.get_tracks(token, tracks_set)
      for i, track_ in enumerate(tracks_set):
        track = profile_tracks[i]
        track['stars'] = track_['stars']
        sp_tracks.append(track)
      offset += 50

    users_data[bardo_id] = sp_tracks

  return users_data

def load_tracks(fname):
  tracks = []
  f = open(fname, 'r')
  for line in f:
    track_info = line.strip().split('\t')
    track = {
      'id': track_info[0],
      'name': track_info[1],
      'stars': int(track_info[2]) if len(track_info) > 2 else -1,
    }
    tracks.append(track)
  f.close()
  return tracks

def load_tracks_to_rate(bardo_id):
  idn = escape_bardo_id(bardo_id)
  plst_dir = f'{ROOT}/{idn}/playlists'
  rated_dir = f'{ROOT}/{idn}/feedback'

  if os.path.isdir(plst_dir):
    plst_tracks = []
    rated_tracks = {}

    for filename in sorted(os.listdir(plst_dir)): 
      if filename.endswith('.txt'):
        for track in load_tracks(os.path.join(plst_dir, filename)):
          plst_tracks.append((track['id'], track['name']))

    if os.path.isdir(rated_dir):
      for filename in os.listdir(rated_dir): 
        if filename.endswith('.txt'):
          for track in load_tracks(os.path.join(rated_dir, filename)):
            rated_tracks[track['id']] = track['name']

    needs_rating = OrderedDict()
    for track in plst_tracks:
      if track[0] not in rated_tracks:
        needs_rating[track[0]] = track[1]
    return needs_rating
  else:
    return {}

def save_playlist(bardo_id, playlist, directory, plst_name):
  idn = escape_bardo_id(bardo_id)
  id_dir = f'{ROOT}/{idn}'
  plst_dir = f'{id_dir}/{directory}'

  if not os.path.isdir(id_dir):
    os.mkdir(id_dir)
  if not os.path.isdir(plst_dir):
    os.mkdir(plst_dir)

  f = open(f'{plst_dir}/{plst_name}.txt', 'w+')
  for i, track in enumerate(playlist['ids']):
    f.write(f'{track}\t{playlist["names"][i]}\n')
  f.close()

def process_plst_feedback(token, url, stars):
  split = 'https://open.spotify.com/playlist/'
  tracks = []
  try:
    parts = url.split(split)
    plst = parts[1].split('?')[0]
    tracks = su.get_playlist(token, plst, f'{stars} feedback')
  except:
    pass

  feedback = []
  for track in tracks:
    feedback.append({
      'id': track['id'],
      'name': track['name'],
      'stars': int(stars),
    })

  return feedback

def process_feedback_input(bardo_id, form):
  needs_rating = load_tracks_to_rate(bardo_id)
  profile = list(load_profile_deduped(bardo_id).values())

  feedback = []
  for track, name in needs_rating.items():
    stars = form.get(f'feedback-{track}')
    if stars:
      stars = int(stars)
      feedback.append({
        'id': track,
        'name': name,
        'stars': stars,
        'extra': json.dumps({
          'size': len(profile),
          '1': len(list(filter(lambda track: track['stars'] == 1, profile))),
          '2': len(list(filter(lambda track: track['stars'] == 2, profile))),
          '3': len(list(filter(lambda track: track['stars'] == 3, profile))),
          '4': len(list(filter(lambda track: track['stars'] == 4, profile))),
          '5': len(list(filter(lambda track: track['stars'] == 5, profile))),
        })
      })

  return feedback

def save_feedback(bardo_id, feedback, directory, name):
  idn = escape_bardo_id(bardo_id)
  id_dir = f'{ROOT}/{idn}'
  feedback_dir = f'{id_dir}/{directory}'

  if not os.path.isdir(id_dir):
    os.mkdir(id_dir)
  if not os.path.isdir(feedback_dir):
    os.mkdir(feedback_dir)

  f = open(f'{feedback_dir}/{name}.txt', 'a+')
  for track in feedback:
    f.write(f'{track["id"]}\t{track["name"]}\t{track["stars"]}')
    if 'extra' in track:
      f.write(f'\t{track["extra"]}')
    f.write(f'\n')
  f.close()

def load_ids():
  bardo_ids = []
  _r, subdirs, _f = next(os.walk(ROOT))
  for subdir in subdirs:
    if subdir.endswith('_com'):
      bardo_ids.append(subdir.replace('-', '@').replace('_', '.'))
  return bardo_ids

def load_users_data(dstart):
  date = datetime.strptime(dstart, '%Y-%m-%d')

  users_data = {}
  for bardo_id in load_ids():
    clf_predictions = {}
    id_dir = escape_bardo_id(bardo_id)
    predict_dir = os.path.join(ROOT, id_dir, 'predictions')
    for pf in os.listdir(predict_dir):
      splits = pf.replace('.txt', '').split('_')
      idx = list(map(
        lambda split: any(c.isalpha() for c in split), 
        splits,
      )).index(True)
      clf = '_'.join(splits[idx:])
      try:
        pdate = datetime.strptime('_'.join(splits[:idx]), '%Y-%m-%d_%H-%M-%S')
        if clf == 'random' or pdate >= date:
          predictions = clf_predictions.get(clf, [])
          clf_predictions[clf] = predictions + load_tracks(
            os.path.join(predict_dir, pf),
          )
      except:
        pass

    users_data[bardo_id] = (load_profile_deduped(bardo_id), clf_predictions)

  return users_data

def load_user_profiles():
  users_data = {}
  for bardo_id in load_ids():
    users_data[bardo_id] = load_profile_deduped(bardo_id).values()
  return users_data

def escape_bardo_id(bardo_id: str) -> str:
  return bardo_id.replace('@', '-').replace('.', '_')
