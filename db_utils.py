import spotify_utils as su
import production_utils as pu
import os

def load_profile(bardo_id):
  idn = bardo_id.replace('@', '-').replace('.', '_')
  rated_dir = f'datasets/{idn}/feedback'
  profile_dir = f'datasets/{idn}/profile'

  profile_files = []
  if os.path.isdir(rated_dir):
    profile_files = profile_files + list(map(
      lambda filename: f'{rated_dir}/{filename}',
      filter(
        lambda filename: filename.endswith('.txt'),
        os.listdir(rated_dir),
      ),
    ))
  if os.path.isdir(profile_dir):
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
    f = open(f'{path}', 'r')
    for line in f:
      track_info = line.strip().split('\t')
      profile.append({
        'id': track_info[0],
        'name': track_info[1],
        'stars': int(track_info[2]),
      })
    f.close()

  return profile

def load_tracks_to_rate(bardo_id):
  idn = bardo_id.replace('@', '-').replace('.', '_')
  id_dir = f'datasets/{idn}'
  plst_dir = f'{id_dir}/playlists'
  rated_dir = f'{id_dir}/feedback'

  if not os.path.isdir(id_dir):
    os.mkdir(id_dir)

  if os.path.isdir(plst_dir):
    plst_tracks = {}
    rated_tracks = {}

    for filename in os.listdir(plst_dir): 
      if filename.endswith('.txt'):
        f = open(f'{plst_dir}/{filename}', 'r')
        for line in f:
          track_info = line.strip().split('\t')
          plst_tracks[track_info[0]] = track_info[1]
        f.close()

    if os.path.isdir(rated_dir):
      for filename in os.listdir(rated_dir): 
        if filename.endswith('.txt'):
          f = open(f'{rated_dir}/{filename}', 'r')
          for line in f:
            track_info = line.strip().split('\t')
            rated_tracks[track_info[0]] = track_info[1]
          f.close()

    needs_rating = {}
    for track, name in plst_tracks.items():
      if track not in rated_tracks and track not in needs_rating:
        needs_rating[track] = name
    return needs_rating
  else:
    return {}

def process_plst_feedback(token, fav_url, no_fav_url):
  split = 'https://open.spotify.com/playlist/'
  fav_tracks = []
  no_fav_tracks = []
  try:
    fav_parts = fav_url.split(split)
    fav_plst = fav_parts[1].split('?')[0]
    fav_tracks = su.get_playlist(token, fav_plst, 'fav')
  except:
    pass
  try:
    no_fav_parts = no_fav_url.split(split)
    no_fav_plst = no_fav_parts[1].split('?')[0]
    no_fav_tracks = su.get_playlist(token, no_fav_plst, 'no fav')
  except:
    pass

  feedback = []
  for track in fav_tracks:
    feedback.append({
      'id': track['id'],
      'name': track['name'],
      'stars': 5,
    })
  for track in no_fav_tracks:
    feedback.append({
      'id': track['id'],
      'name': track['name'],
      'stars': 2,
    })

  return feedback

def process_feedback_input(needs_rating, form):
  feedback = []
  for track, name in needs_rating.items():
    stars = 4
    fval = form.get(f'feedback-{track}')
    if fval == 'like':
      stars = 5
    elif fval == 'no-like':
      stars = 2
    feedback.append({
      'id': track,
      'name': name,
      'stars': stars,
    })

  return feedback

def save_feedback(bardo_id, feedback, d, name):
  idn = bardo_id.replace('@', '-').replace('.', '_')
  id_dir = f'datasets/{idn}'
  feedback_dir = f'{id_dir}/{d}'

  if not os.path.isdir(id_dir):
    os.mkdir(id_dir)
  if not os.path.isdir(feedback_dir):
    os.mkdir(feedback_dir)

  f = open(f'{feedback_dir}/{name}.txt', 'w+')
  for track in feedback:
    f.write(f'{track["id"]}\t{track["name"]}\t{track["stars"]}\n')
  f.close()
