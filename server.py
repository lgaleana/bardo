from flask import Flask, request, redirect, url_for
import os
import time
import bard.spotify_utils as su
import bard.production_utils as pu

app = Flask(__name__)

CLIENT_ID = '8de267b03c464274a3546bfe84496696'

classifiers = pu.load_prod_classifiers()

def make_playlists(token):
  playlists = pu.generate_recommendations(
    token,
    ['house', 'techno'],
    classifiers,
    10,
  )
  for name, tracks in playlists:
    playlist_name = f'{name}_{time.time()}'.replace('.', '')
    playlist = su.create_playlist(token, playlist_name)
    su.populate_playlist(token, playlist, tracks)


@app.route('/')
def main():
  return f'''
  <H1>Bard: the AI Music Curator</h1>
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
  <h1>Bard: the AI Music Curator</h1>
  <h2>Playlist Generation</h2>
  {html}
  '''
