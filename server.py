from flask import Flask, request, redirect, url_for, render_template
import spotify_utils as su
import production_utils as pu
from datetime import datetime
import os

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

def load_tracks_to_rate(bardo_id):
  data_dir = 'datasets'
  idn = bardo_id.replace('@', '-').replace('.', '_')
  id_dir = f'{data_dir}/{idn}'
  plst_dir = f'{id_dir}/playlists'
  rated_dir = f'{id_dir}/rated'

  if not os.path.isdir(id_dir):
    os.mkdir(id_dir)
  if not os.path.isdir(rated_dir):
    os.mkdir(rated_dir)

  if os.path.isdir(plst_dir):
    track_infos = []
    rated_tracks = []

    for filename in os.listdir(plst_dir): 
      if filename.endswith('.txt'):
        f = open(f'{plst_dir}/{filename}', 'r')
        for line in f:
          track_info = line.strip().split('\t')
          track_infos.append(track_info)
    for filename in os.listdir(rated_dir): 
      f = open(f'{rated_dir}/{filename}', 'r')
      for line in f:
        track = line.strip().split('\t')[0]
        rated_tracks.append(track)

    needs_rating = {}
    for info in track_infos:
      if info[0] not in rated_tracks and info[0] not in needs_rating:
        needs_rating[info[0]] = 0
#        needs_rating[info[0]] = info[1]
    return needs_rating
  else:
    return {}

@app.route('/')
def main():
  return render_template(
    'index.html',
    playlist_url=url_for('generate_playlist'),
    rate_url=url_for('rate_recommendations'),
  )

@app.route('/identify')
def identify():
  redirect_url = request.args.get('redirect-url')
  if not redirect_url:
    redirect_url = url_for('main')
  return render_template(
    'identify.html',
    redirect_url=redirect_url,
  )

@app.route('/generate-playlist')
def generate_playlist():
  bardo_id = request.args.get('bardo-id')
  if bardo_id:
    needs_rating = load_tracks_to_rate(bardo_id)
    if len(needs_rating) == 0:
      code = request.args.get('code')
      if code:
        token = su.request_token(
          'authorization_code',
          code,
          url_for('generate_playlist', _external=True),
        )
        make_playlists(token)
        return render_template('generate-playlist.html')
      else:
        redirect_uri = url_for('generate_playlist', _external=True)
        return redirect(
          f'https://accounts.spotify.com/authorize?client_id={CLIENT_ID}&response_type=code&redirect_uri={redirect_uri}&scope=playlist-modify-public playlist-modify-private&show_dialog=true',
        )
    else:
      generate_url = url_for("generate_playlist").replace('/', '')
      return redirect(
        f'{url_for("rate_recommendations")}?bardo-id={bardo_id}&redirect-url={generate_url}'
      )
  else:
    generate_url = url_for("generate_playlist").replace('/', '')
    return redirect(
      f'{url_for("identify")}?redirect-url={generate_url}'
    )

@app.route('/rate-recommendations')
def rate_recommendations():
  redirect_url = request.args.get('redirect-url')
  if not redirect_url:
    redirect_url = url_for('main')

  bardo_id = request.args.get('bardo-id')
  if bardo_id:
    needs_rating = load_tracks_to_rate(bardo_id)
    save_url = url_for('save_ratings')
    redirect_url = redirect_url.replace('/', '')
    return render_template(
      'rate-recommendations.html',
      needs_rating=needs_rating,
      save_url=f'{save_url}?bardo-id={bardo_id}&redirect-url={redirect_url}',
    )
  else:
    rate_url = url_for("rate_recommendations").replace('/', '')
    return redirect(
      f'{url_for("identify")}?redirect-url={rate_url}'
    )

@app.route('/save-ratings', methods=['POST'])
def save_ratings():
  return ''
