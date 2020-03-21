import requests
import base64
import spotify_utils as su

playlists = [
  '2Ekqf5nF5x27sRuV0QNqCY',
  '6iX8Jb3FH6vHkx1gMGIH1V',
  '3pa9BxfAiNIrpIA50JYE21',
  '4Tm0k4YEeXTZB6HPADloUM',
  '6fvR540fhVMMwQ3iwaD6aD',
]
tracks = [] 

PLAYLIST_URL = 'https://api.spotify.com/v1/playlists/{}/tracks?limit=100&offset={}'

def get_playlist(token, playlist_id, stars):
  headers = {'Authorization': f'Bearer {token}'}
  offset = 0
  playlist = []

  while True:
    print(f'Getting playlist {i}-stars, offset {offset}')
    url = PLAYLIST_URL.format(playlist_id, offset)
    r = requests.get(url, headers=headers)
    items = r.json()['items']
    for item in items:
      playlist.append(item['track'])
    if len(items) == 0:
      break;
    offset += 100
  return playlist


token = su.request_token()
print(f'Access token: {token}')

print('Retrieving samples')
for i, playlist_id in enumerate(playlists, start=1):
  playlist = get_playlist(token, playlist_id, i)
  for track in playlist:
    track['stars'] = i
    tracks.append(track)
print()

dataset = open('dataset.txt', 'w')
tracks_db = open('tracks.txt', 'w')
print('Writing features')
for track in tracks:
  tracks_db.write(f'{track["id"]}\n')
  tracks_db.flush()

  track_features = su.get_track_features(token, track)
  features = map(lambda feature: str(feature), track_features)
  features_str = ','.join(features)

  dataset.write(f'{features_str},{track["stars"]}\n')
  dataset.flush()

tracks_db.close()
dataset.close()

print('Dataset generated: {} samples'.format(len(tracks)))
