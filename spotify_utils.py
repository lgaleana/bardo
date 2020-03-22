import requests
import base64
import json

CLIENT_ID = '8de267b03c464274a3546bfe84496696'
CLIENT_SECRET = '11ac7098025545af90be092fe2dd029c'

AUTHORIZATION_URL = 'https://accounts.spotify.com/api/token'
FEATURES_URL = 'https://api.spotify.com/v1/audio-features'
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

def get_track_features(token, track):
  print(f'Getting track features for {track["name"]}')
  url = f'{FEATURES_URL}/{track["id"]}'
  headers = {'Authorization': f'Bearer {token}'}

  r = requests.get(url, headers=headers)
  track_features = r.json()

  useful_features = [
    track['popularity'],
    track_features['duration_ms'],
    track_features['time_signature'],
    track_features['acousticness'],
    track_features['key'],
    track_features['danceability'],
    track_features['energy'],
    track_features['instrumentalness'],
    track_features['liveness'],
    track_features['mode'],
    track_features['loudness'],
    track_features['speechiness'],
    track_features['valence'],
    track_features['tempo'],
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
