from flask import Flask, request, redirect, url_for, render_template
import production_utils as pu
import db_utils as db
import spotify_utils as su
from datetime import datetime

app = Flask(__name__)

CLIENT_ID = '8de267b03c464274a3546bfe84496696'
EXP_CONFIG = ['gbdt_bottom_high', 'random']
PLAYLIST_LIMIT = 10
TIME_LIMIT = 300
post_auth = 'main'


pu.load_prod_classifiers()

@app.route('/')
def main():
  return render_template('index.html')

@app.route('/playlist-selection')
def playlist_selection():
  bardo_id = request.args.get('bardo-id')
  token = request.args.get('token')
  if token:
    return render_template(
      'playlist-selection.html',
      bardo_id=bardo_id if bardo_id else '',
      token=token,
      exp_clfs=','.join(EXP_CONFIG),
    )
  else:
    global post_auth
    post_auth = 'playlist_selection'
    return redirect(f'https://accounts.spotify.com/authorize?client_id={CLIENT_ID}&response_type=code&redirect_uri={url_for("spotify_auth", _external=True)}&scope=playlist-modify-public playlist-modify-private')

@app.route('/generate-playlist')
def generate_playlist():
  bardo_id = request.args.get('bardo-id')
  token = request.args.get('token')
  genre = request.args.get('genre')
  source = request.args.get('source')
  market = request.args.get('market')

  if token and genre and source:
    if bardo_id:
      needs_rating = db.load_tracks_to_rate(bardo_id)
      if len(needs_rating) == 0:
        return render_template(
          'generate-playlist.html',
          bardo_id=bardo_id,
          data={'token': token, 'genre': genre, 'source': source},
        )
      else:
        rurl = url_for('rate_recommendations')
        return redirect(
          f'{rurl}?bardo-id={bardo_id}&token={token}&genre={genre}&source={source}&redirect-uri=generate_playlist'
        )
    else:
      iurl = url_for('identify')
      return redirect(
        f'{iurl}?token={token}&genre={genre}&source={source}&redirect-uri=generate_playlist'
      )
  else:
    return 'Invalid request.'

@app.route('/make-playlist/<bardo_id>', methods=['POST'])
def make_playlist(bardo_id):
  token = request.json.get('token')
  genre = request.json.get('genre')
  source = request.json.get('source')
  market = request.json.get('market')

  if token and source and genre:
    playlists = pu.gen_recs(
      token,
      genre.split(','),
      source.split(','),
      'MX',
      PLAYLIST_LIMIT,
      TIME_LIMIT,
    )
    now = datetime.now().strftime("%d-%m-%Y_%H-%M")
    db.save_playlists(bardo_id, playlists, now)

    final_plst = playlists['final']
    if len(final_plst) > 0:
      playlist = su.create_playlist(token, f'Bardo {now}')
      su.populate_playlist(token, playlist, final_plst['ids'])
      return f'Playlist <b>{playlist["name"]}</b> has been created in your spotify account.'
    else:
      return 'No tracks were generated. Please try again.'
  else:
    return 'Invalid request.'

@app.route('/profile')
def profile():
  bardo_id = request.args.get('bardo-id')
  if bardo_id:
    needs_rating = db.load_tracks_to_rate(bardo_id)
    if len(needs_rating) == 0:
      token = request.args.get('token')
      if token:
        return render_template(
          'profile.html',
          bardo_id=bardo_id,
          token=token,
        )
      else:
        global post_auth
        post_auth = 'profile'
        return redirect(f'https://accounts.spotify.com/authorize?client_id={CLIENT_ID}&response_type=code&redirect_uri={url_for("spotify_auth", _external=True)}&scope=playlist-modify-public playlist-modify-private')
    else:
      rurl = url_for('rate_recommendations')
      return redirect(f'{rurl}?bardo-id={bardo_id}&redirect-uri=profile')
  else:
    iurl =url_for('identify') 
    return redirect(f'{iurl}?redirect-uri=profile')

@app.route('/tracks/<bardo_id>/<label>')
def tracks(bardo_id, label):
  profile = db.load_profile(bardo_id)
  if label == 'liked':
    profile = filter(lambda track: track['stars'] >= 5, profile)
  elif label == 'not-liked':
    profile = filter(lambda track: track['stars'] <= 2, profile)
  elif label != 'all':
    return 'Invalid request.'

  return render_template('tracks.html', tracks=profile)

@app.route('/save-playlists/<bardo_id>', methods=['POST'])
def save_playlists(bardo_id):
  token = request.args.get('token')
  favorite_url = request.form.get('favorite')
  not_favorite_url = request.form.get('no-favorite')

  if token and (favorite_url or not_favorite_url):
    feedback = db.process_plst_feedback(
      token,
      favorite_url,
      not_favorite_url,
    )
    if len(feedback) > 0:
      now = datetime.now().strftime("%d-%m-%Y")
      db.save_feedback(bardo_id, feedback, 'profile', now)
      return 'Tracks saved.'
    else:
      return 'No tracks saved.'
  else:
    return 'Invalid request.'

@app.route('/identify')
def identify():
  redirect_uri = request.args.get('redirect-uri')
  if not redirect_uri:
    redirect_uri = 'main'
  args = filter(lambda arg: arg[0] != 'redirect-uri', request.args.items())
  return render_template(
    'identify.html',
    redirect_url=redirect_uri,
    args=args,
  )

@app.route('/rate-recommendations')
def rate_recommendations():
  bardo_id = request.args.get('bardo-id')
  if bardo_id:
    needs_rating = db.load_tracks_to_rate(bardo_id)
    save_url = url_for('save_ratings', bardo_id=bardo_id)
    save_url += '?' + get_request_params(request.args)
    return render_template(
      'rate-recommendations.html',
      needs_rating=needs_rating,
      save_url=f'{save_url}',
    )
  else:
    iurl = url_for('identify')
    return redirect(
      f'{iurl}?redirect-uri=rate_recommendations'
    )

@app.route('/save-ratings/<bardo_id>', methods=['POST'])
def save_ratings(bardo_id):
  redirect_uri = request.args.get('redirect-uri')

  now = datetime.now().strftime("%d-%m-%Y_%H-%M")
  db.save_feedback(bardo_id, db.process_feedback_input(
    db.load_tracks_to_rate(bardo_id),
    request.form,
  ), 'feedback', now)

  if redirect_uri:
    redirect_url = url_for(redirect_uri)
    redirect_url += '?' + get_request_params(request.args, 'redirect-uri')
    return redirect(f'{redirect_url}')
  else:
    return 'Tracks saved.'

@app.route('/spotify-auth')
def spotify_auth():
  code = request.args.get('code')
  if code:
    token = su.request_token(
      'authorization_code',
      code,
      url_for("spotify_auth", _external=True),
    )
    return render_template(
      'spotify-auth.html',
      token=token,
      redirect_url=url_for(post_auth),
    )
  else:
    return 'Invalid request'

def get_request_params(request, exclude=''):
  params = ''
  for name, param in request.items():
    if name != exclude:
      params += f'{name}={param}&'
  return params
