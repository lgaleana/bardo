import requests
import base64
import json
import numpy as np

CLIENT_ID = '8de267b03c464274a3546bfe84496696'
CLIENT_SECRET = '11ac7098025545af90be092fe2dd029c'

AUTHORIZATION_URL = 'https://accounts.spotify.com/api/token'
PLAYLIST_URL = 'https://api.spotify.com/v1/playlists/{}/tracks?limit=100&offset={}'
FEATURES_URL = 'https://api.spotify.com/v1/audio-features'
ANALYSIS_URL = 'https://api.spotify.com/v1/audio-analysis'
RECS_URL = 'https://api.spotify.com/v1/recommendations?limit=100'
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

def get_playlist(token, playlist_id, name):
  headers = {'Authorization': f'Bearer {token}'}
  offset = 0
  playlist = []

  while True:
    print(f'Getting playlist {name}, offset {offset}')
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
  useful_features = useful_features + [
    len(bars),
    np.mean(durations),
    np.std(durations),
    np.var(durations),
  ]
  # Beats
  beats = analysis['beats']
  durations = list(map(lambda beat: beat['duration'], beats))
  useful_features = useful_features + [
    len(beats),
    np.mean(durations),
    np.std(durations),
    np.var(durations),
  ]
  # Sections 
  sections = analysis['sections']
  durations = []
  loudness = []
  tempos = []
  modes = []
  ts = []
  for section in sections:
    durations.append(section['duration'])
    loudness.append(section['loudness'])
    tempos.append(section['tempo'])
    modes.append(section['mode'])
    ts.append(section['time_signature'])
  useful_features = useful_features + [
    len(sections),
    np.mean(durations),
    np.std(durations),
    np.var(durations),
    np.mean(loudness),
    np.std(loudness),
    np.var(loudness),
    np.mean(tempos),
    np.std(tempos),
    np.var(tempos),
    np.mean(modes),
    np.std(modes),
    np.var(modes),
    np.mean(ts),
    np.std(ts),
    np.var(ts),
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
  useful_features = useful_features + [
    len(segments),
    np.mean(durations),
    np.std(durations),
    np.var(durations),
    np.mean(ls),
    np.std(ls),
    np.var(ls),
    np.mean(lmt),
    np.std(lmt),
    np.var(lmt),
    np.mean(lm),
    np.std(lm),
    np.var(lm),
  ]
  useful_features = useful_features + np.mean(pitches, axis=0).tolist() + np.mean(timbre, axis=0).tolist()
# Tatums
  tatums = analysis['tatums']
  durations = list(map(lambda tatum: tatum['duration'], tatums))
  useful_features = useful_features + [
    len(tatums),
    np.mean(durations),
    np.std(durations),
    np.var(durations),
  ]

  return useful_features

def get_recommendations(token, seeds, market=''):
  sparams = []
  for name, seed in seeds.items():
    sparams.append(f'seed_{name}={",".join(seed)}')
  params = ''
  if len(sparams) > 0:
    params = f'&{"&".join(sparams)}'
  if market:
    params += f'&market={market}'
  url = f'{RECS_URL}{params}'
  print(f'Getting recommendations: {url}')
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
