from flask import Flask, request, redirect, url_for
import bard.spotify_utils as su

app = Flask(__name__)

CLIENT_ID = '8de267b03c464274a3546bfe84496696'

@app.route('/')
def main():
  return f'''
  <H1>Bard: the AI Music Curator</h1>
  <a href='https://accounts.spotify.com/authorize?client_id={CLIENT_ID}&response_type=code&redirect_uri={url_for('generate_playlist', _external=True)}&scope=playlist-modify-public playlist-modify-private'>Generate Playlist</a>
  '''

@app.route('/generate-playlist')
def generate_playlist():
  code = request.args.get('code')
  if code:
    token = su.request_token(code)
    html = 'Playlist generated'
  else:
    html = '<p>Invalid access</p>'

  return f'''
  <h1>Bard: the AI Music Curator</h1>
  <h3>Playlist Generation</h3>
  {html}
  '''
