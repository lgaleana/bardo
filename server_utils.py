import spotify_utils as su
import production_utils as pu
from datetime import datetime
import os

def make_playlist(code, redirect_url):
  token = su.request_token(
    'authorization_code',
    code,
    redirect_url,
  )

  now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
  tracks = pu.generate_recommendations(
    token,
    ['deep-house'],
    PLAYLIST_LIMIT,
    now,
  )

  playlist = su.create_playlist(token, f'{now}')
  su.populate_playlist(token, playlist, tracks)

def load_tracks_to_rate(bardo_id):
  idn = bardo_id.replace('@', '-').replace('.', '_')
  id_dir = f'datasets/{idn}'
  plst_dir = f'{id_dir}/playlists'
  rated_dir = f'{id_dir}/feedback'

  if not os.path.isdir(id_dir):
    os.mkdir(id_dir)

  if os.path.isdir(plst_dir):
    track_infos = []
    rated_tracks = []
    for filename in os.listdir(plst_dir): 
      if filename.endswith('.txt'):
        f = open(f'{plst_dir}/{filename}', 'r')
        for line in f:
          track_info = line.strip().split('\t')
          track_infos.append(track_info)
    if os.path.isdir(rated_dir):
      for filename in os.listdir(rated_dir): 
        if filename.endswith('.txt'):
          f = open(f'{rated_dir}/{filename}', 'r')
          for line in f:
            track = line.strip().split('\t')[0]
            rated_tracks.append(track)

    needs_rating = {}
    for info in track_infos:
      if info[0] not in rated_tracks and info[0] not in needs_rating:
        needs_rating[info[0]] = info[1]
    return needs_rating
  else:
    return {}

def save_feedback(bardo_id, likes, no_likes, neutral=[]):
  likes = set(likes)
  no_likes = set(no_likes)
  neutral = set(neutral)
  neutral = neutral - likes
  neutral = neutral - no_likes
  
  idn = bardo_id.replace('@', '-').replace('.', '_')
  id_dir = f'datasets/{idn}'
  rated_dir = f'{id_dir}/feedback'

  if not os.path.isdir(rated_dir):
    os.mkdir(rated_dir)

  now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
  f = open(f'{rated_dir}/{now}.txt', 'w+')
  for track in no_likes:
    f.write(f'{track}\t2\n')
  for track in neutral:
    f.write(f'{track}\t4\n')
  for track in likes:
    f.write(f'{track}\t5\n')
  f.close()
