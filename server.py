from flask import Flask, request, redirect, url_for
import time
import spotify_utils as su
import production_utils as pu
from datetime import datetime

app = Flask(__name__)

CLIENT_ID = '8de267b03c464274a3546bfe84496696'

PLAYLIST_LIMIT = 20
pu.load_prod_classifiers()

def make_playlists(token):
  now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
  tracks = pu.generate_recommendations(
    token,
    ['deep-house'],
    PLAYLIST_LIMIT,
    now,
  )
  playlist = su.create_playlist(token, f'{now}')
  su.populate_playlist(token, playlist, tracks)


@app.route('/')
def main():
  return f'''
  <H1>Bard: AI Deep House Curator (MVP)</h1>
  <a href='https://accounts.spotify.com/authorize?client_id={CLIENT_ID}&response_type=code&redirect_uri={url_for('generate_playlist', _external=True)}&scope=playlist-modify-public playlist-modify-private&show_dialog=true'>Generate Playlist</a>
  '''

@app.route('/generate-playlist')
def generate_playlist():
  code = request.args.get('code')
  if code:
    token = su.request_token(
      'authorization_code',
      code,
      url_for('generate_playlist', _external=True),
    )
    make_playlists(token)
    html = 'Playlist generated'
  else:
    html = '<p>Invalid access</p>'

  return f'''
  <h1>Bard: Deep House Curator (MVP)</h1>
  {html}
  '''
