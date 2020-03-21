import requests
import base64

CLIENT_ID = '8de267b03c464274a3546bfe84496696'
CLIENT_SECRET = '11ac7098025545af90be092fe2dd029c'

AUTHORIZATION_URL = 'https://accounts.spotify.com/api/token'
FEATURES_URL = 'https://api.spotify.com/v1/audio-features'

def request_token():
  authorization_data = f'{CLIENT_ID}:{CLIENT_SECRET}'.encode('utf-8')
  parameters = {'grant_type': 'client_credentials'}
  headers = {'Authorization': 'Basic {}'.format(
    base64.b64encode(authorization_data).decode()
  )}

  print('Requesting token')
  r = requests.post(AUTHORIZATION_URL, data=parameters, headers=headers)
  return r.json()['access_token']

def get_track_features(token, track):
  url = f'{FEATURES_URL}/{track["id"]}'
  headers = {'Authorization': f'Bearer {token}'}

  print(f'Getting track features for {track["name"]}')
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
  url = f'https://api.spotify.com/v1/recommendations?limit=100&seed_genres={genres_str}'
  headers = {'Authorization': f'Bearer {token}'}
  r = requests.get(url, headers=headers)
  return r.json()['tracks']
