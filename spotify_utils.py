import requests
import base64
import json
import statistics as s

CLIENT_ID = '8de267b03c464274a3546bfe84496696'
CLIENT_SECRET = '11ac7098025545af90be092fe2dd029c'

AUTHORIZATION_URL = 'https://accounts.spotify.com/api/token'
PLAYLIST_URL = 'https://api.spotify.com/v1/playlists/{}/tracks?limit=100&offset={}'
FEATURES_URL = 'https://api.spotify.com/v1/audio-features'
ANALYSIS_URL = 'https://api.spotify.com/v1/audio-analysis'
CREATE_PLAYLIST_URL = 'https://api.spotify.com/v1/users/lsgaleana/playlists'
POPULATE_PLAYLIST_URL = 'https://api.spotify.com/v1/playlists/{}/tracks?uris={}'

def request_token(grant_type='client_credentials', code=None, redirect_uri=None):
  print('Requesting token')
  authorization_data = f'{CLIENT_ID}:{CLIENT_SECRET}'.encode('utf-8')
  parameters = {'grant_type': grant_type}
  headers = {'Authorization': 'Basic {}'.format(
    base64.b64encode(authorization_data).decode()
  )}
  if code is not None:
    parameters['code'] = code
  if redirect_uri is not None:
    parameters['redirect_uri'] = redirect_uri

  r = requests.post(AUTHORIZATION_URL, data=parameters, headers=headers)
  return r.json()['access_token']

def get_playlist(token, playlist_id, stars):
  headers = {'Authorization': f'Bearer {token}'}
  offset = 0
  playlist = []

  while True:
    print(f'Getting playlist {stars}-stars, offset {offset}')
    url = PLAYLIST_URL.format(playlist_id, offset)
    r = requests.get(url, headers=headers)
    items = r.json()['items']
    for item in items:
      playlist.append(item['track'])
    if len(items) == 0:
      break;
    offset += 100
  return playlist

def get_tracks_features(token, tracks):
  print(f'Getting audio features for {len(tracks)} tracks')
  track_ids = ','.join(map(
    lambda track: track['id'],
    tracks,
  ))
  url = f'{FEATURES_URL}?ids={track_ids}'
  headers = {'Authorization': f'Bearer {token}'}

  r = requests.get(url, headers=headers)
  audio_features = r.json()['audio_features']

  useful_features = []
  for i, track in enumerate(tracks):
    features = audio_features[i]
    useful_features.append([
      track['popularity'],
      features['duration_ms'],
      features['time_signature'],
      features['acousticness'],
      features['key'],
      features['danceability'],
      features['energy'],
      features['instrumentalness'],
      features['liveness'],
      features['mode'],
      features['loudness'],
      features['speechiness'],
      features['valence'],
      features['tempo'],
    ])

  return useful_features 

def get_track_analysis(token, track):
  print(f'Getting track analysis for {track["name"]}')
  url = f'{ANALYSIS_URL}/{track["id"]}'
  headers = {'Authorization': f'Bearer {token}'}

  r = requests.get(url, headers=headers)
  analysis = r.json()

  useful_features = []
  # Track features
  track = analysis['track']
  useful_features = useful_features + [
    track['offset_seconds'],
    track['end_of_fade_in'],
    track['start_of_fade_out'],
  ]
  # Bars
  bars = analysis['bars']
  durations = list(map(lambda bar: bar['duration'], bars))
  mean = s.mean(durations)
  useful_features = useful_features + [
    len(bars),
    mean,
    s.stdev(durations, mean),
    s.variance(durations, mean),
  ]
  # Beats
  beats = analysis['beats']
  durations = list(map(lambda beat: beat['duration'], beats))
  mean = s.mean(durations)
  useful_features = useful_features + [
    len(beats),
    mean,
    s.stdev(durations, mean),
    s.variance(durations, mean),
  ]
  # Sections 
  sections = analysis['sections']
  durations = []
  loudness = []
  tempos = []
  keys = []
  modes = []
  ts = []
  for section in sections:
    durations.append(section['duration'])
    loudness.append(section['loudness'])
    tempos.append(section['tempo'])
    keys.append(section['key'])
    modes.append(section['mode'])
    ts.append(section['time_signature'])
  d_mean = s.mean(durations)
  l_mean = s.mean(loudness)
  t_mean = s.mean(tempos)
  k_mean = s.mean(keys)
  m_mean = s.mean(modes)
  ts_mean = s.mean(ts)
  useful_features = useful_features + [
    len(sections),
    d_mean,
    s.stdev(durations, d_mean),
    s.variance(durations, d_mean),
    l_mean,
    s.stdev(loudness, l_mean),
    s.variance(loudness, l_mean),
    t_mean,
    s.stdev(tempos, t_mean),
    s.variance(tempos, t_mean),
    k_mean,
    s.stdev(keys, k_mean),
    s.variance(keys, k_mean),
    m_mean,
    s.stdev(modes, m_mean),
    s.variance(modes, m_mean),
    ts_mean,
    s.stdev(ts, ts_mean),
    s.variance(ts,ts_mean),
  ]
  # Segments
  segments = analysis['segments']
  durations = []
  ls = []
  lmt = []
  lm = []
  pitches = []
  timbre = []
  for segment in segments:
    durations.append(segment['duration'])
    ls.append(segment['loudness_start'])
    lmt.append(segment['loudness_max_time'])
    lm.append(segment['loudness_max'])
    pitches.append(segment['pitches'])
    timbre.append(segment['timbre'])
  d_mean = s.mean(durations)
  ls_mean = s.mean(ls)
  lmt_mean = s.mean(lmt)
  lm_mean = s.mean(lm)

  pitches_mean = []
  pitches_stdev = []
  pitches_var = []
  for pitch in pitches:
    pitch_mean = s.mean(pitch)
    pitches_mean.append(pitch_mean)
    pitches_stdev.append(s.stdev(pitch, pitch_mean))
    pitches_var.append(s.variance(pitch, pitch_mean))
  timbre_mean = []
  timbre_stdev = []
  timbre_var = []
  for timb in timbre:
    timb_mean = s.mean(timb)
    timbre_mean.append(timb_mean)
    timbre_stdev.append(s.stdev(timb, timb_mean))
    timbre_var.append(s.variance(timb, timb_mean))
  useful_features = useful_features + [
    len(segments),
    d_mean,
    s.stdev(durations, d_mean),
    s.variance(durations, d_mean),
    ls_mean,
    s.stdev(ls, ls_mean),
    s.variance(ls, ls_mean),
    lmt_mean,
    s.stdev(lmt, lmt_mean),
    s.variance(lmt, lmt_mean),
    lm_mean,
    s.stdev(lm, lm_mean),
    s.variance(lm, lm_mean),
    s.mean(pitches_mean),
    s.mean(pitches_stdev),
    s.mean(pitches_var),
    s.mean(timbre_mean),
    s.mean(timbre_stdev),
    s.mean(timbre_var),
  ]
# Tatums
  tatums = analysis['tatums']
  durations = list(map(lambda tatum: tatum['duration'], tatums))
  mean = s.mean(durations)
  useful_features = useful_features + [
    len(tatums),
    mean,
    s.stdev(durations, mean),
    s.variance(durations, mean),
  ]

  return useful_features

def get_recommendations(token, genres):
  genres_str = ','.join(genres)
  print(f'Getting {genres_str} recommendations')
  url = f'https://api.spotify.com/v1/recommendations?limit=100&market=MX&seed_genres={genres_str}'
  headers = {'Authorization': f'Bearer {token}'}
  r = requests.get(url, headers=headers)
  return r.json()['tracks']

def create_playlist(token, name):
  print(f'Creating playlist {name}')
  data = json.dumps({'name': name})
  headers = {
    'Authorization': f'Bearer {token}',
    'Accept': 'application/json',
    'Content-Type': 'application/json',
  }

  r = requests.post(CREATE_PLAYLIST_URL, data=data, headers=headers)
  return r.json()

def populate_playlist(token, playlist, tracks):
  print(f'Populating playlist {playlist["name"]}')
  formatted_tracks = map(lambda track: f'spotify:track:{track}', tracks)
  url = POPULATE_PLAYLIST_URL.format(playlist['id'], ','.join(formatted_tracks))
  headers = {'Authorization': f'Bearer {token}'}

  r = requests.post(url, headers=headers)
