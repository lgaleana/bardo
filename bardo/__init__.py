from flask import Flask, request, redirect, url_for, render_template
import bardo.utils.production_utils as pu
import bardo.utils.db_utils as db
import bardo.utils.spotify_utils as su
from datetime import datetime
import logging
from werkzeug.exceptions import InternalServerError

CLIENT_ID = '8de267b03c464274a3546bfe84496696'
EXP_CONFIG = ['gbdt_bottom_high', 'random']
PLAYLIST_LIMIT = 10
TIME_LIMIT = 300
INVALID_REQUEST = '<meta name="viewport" content="width=device-width">Invalid request.'
post_auth = 'main'


app = Flask(__name__, instance_relative_config=True)
logging.basicConfig(filename='error.log', level=logging.ERROR)
pu.load_prod_classifiers()

@app.route('/')
def main():
  return render_template('index.html')

@app.route('/playlist-selection')
def playlist_selection():
  def response(token, bardo_id):
    return render_template(
      'playlist-selection.html',
      token=token,
      exp_clfs=','.join(EXP_CONFIG),
    )
  return validate_response(
    response,
    request,
    'playlist_selection',
    check_recs=False,
  )

@app.route('/generate-playlist')
def generate_playlist():
  genre = request.args.get('genre')
  source = request.args.get('source')
  market = request.args.get('market')

  if not genre:
    genre = 'deep-house'
  if not source:
    source = 'bardo'
  if not market:
    market = 'US'

  def response(token, bardo_id, genre, source, market):
    return render_template(
      'generate-playlist.html',
      bardo_id=bardo_id,
      data={
        'token': token,
        'genre': genre,
        'source': source,
        'market': market,
      },
    )

  return validate_response(
    response,
    request,
    'generate_playlist',
    genre=genre,
    source=source,
    market=market,
  )

@app.route('/make-playlist', methods=['POST'])
def make_playlist():
  token = request.json.get('token')
  genre = request.json.get('genre')
  source = request.json.get('source')
  market = request.json.get('market')

  if not token and not genre and not source and not market:
    return INVALID_REQUEST

  sid, bardo_id = su.get_user_data(token)
  needs_rating = db.load_tracks_to_rate(bardo_id)
  if len(needs_rating) > 0:
    return 'Please first rate previous recommendations.'

  clf_plsts, final_plst = pu.gen_recs(
    token,
    genre.split(','),
    source.split(','),
    market,
    db.load_profile(bardo_id),
    PLAYLIST_LIMIT,
    TIME_LIMIT,
  )

  now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
  db.save_playlist(bardo_id, final_plst, 'playlists', now)
  for clf, plst in clf_plsts.items():
    db.save_playlist(bardo_id, plst, 'predictions', f'{now}_{clf}')

  if len(final_plst) > 0:
    playlist = su.create_playlist(token, sid, f'Bardo {now}')
    su.populate_playlist(token, playlist, final_plst['ids'])
    return f'Playlist <b>{playlist["name"]}</b> has been created in your spotify account.'
  else:
    return 'No tracks were generated. Please try again.'

@app.route('/profile')
def profile():
  def response(token, bardo_id):
    return render_template(
      'profile.html',
      token=token,
      bardo_id=bardo_id,
    )
  return validate_response(response, request, 'profile', check_recs=False)

@app.route('/tracks/<bardo_id>/<stars>')
def tracks(bardo_id, stars):
  stars = int(stars)
  profile = db.load_profile(bardo_id)
  profile = filter(lambda track: track['stars'] == stars, profile)

  return render_template('tracks.html', tracks=profile)

@app.route('/save-playlists/<bardo_id>', methods=['POST'])
def save_playlists(bardo_id):
  token = request.args.get('token')
  url = request.form.get('url')
  stars = request.form.get("feedback")

  if not token and not url and not stars:
    return INVALID_REQUEST

  feedback = db.process_plst_feedback(
    token,
    url,
    stars,
  )
  if len(feedback) > 0:
    now = datetime.now().strftime("%d-%m-%Y")
    db.save_feedback(bardo_id, feedback, 'profile', f'{stars}_{now}')
    return '<meta name="viewport" content="width=device-width">Tracks saved.'
  else:
    return '<meta name="viewport" content="width=device-width">No tracks saved.'

@app.route('/how-it-works')
def how_it_works():
  return render_template('how-it-works.html')

@app.route('/rate-recommendations')
def rate_recommendations():
  def response(token, bardo_id):
    needs_rating = db.load_tracks_to_rate(bardo_id)
    save_url = url_for('save_ratings', bardo_id=bardo_id)
    save_url += '?' + get_request_params(request.args)
    return render_template(
      'rate-recommendations.html',
      needs_rating=needs_rating,
      save_url=f'{save_url}',
    )

  bardo_id = request.args.get('bardo-id')
  if bardo_id:
    return response(None, bardo_id)
  else:
    return validate_response(
      response,
      request,
      'rate_recommendations',
      check_recs=False,
    )

@app.route('/save-ratings/<bardo_id>', methods=['POST'])
def save_ratings(bardo_id):
  redirect_uri = request.args.get('redirect-uri')

  now = datetime.now().strftime("%d-%m-%Y")
  db.save_feedback(bardo_id, db.process_feedback_input(
    db.load_tracks_to_rate(bardo_id),
    request.form,
  ), 'feedback', now)

  if redirect_uri:
    redirect_url = url_for(redirect_uri)
    redirect_url += '?' + get_request_params(request.args, 'redirect-uri')
    return redirect(f'{redirect_url}')
  else:
    return '<meta name="viewport" content="width=device-width">Tracks saved.'

@app.route('/spotify-auth')
def spotify_auth():
  code = request.args.get('code')

  if not code:
    return INVALID_REQUEST

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

@app.errorhandler(InternalServerError)
def handle_500(e):
  app.logger.error(e)
  return '<meta name="viewport" content="width=device-width">There was an error with the application.'


def validate_response(
  response,
  request,
  redirect_uri,
  check_recs=True,
  **kwargs,
):
  token = request.args.get('token')
  if not token:
    global post_auth
    post_auth = redirect_uri
    return redirect(f'https://accounts.spotify.com/authorize?client_id={CLIENT_ID}&response_type=code&redirect_uri={url_for("spotify_auth", _external=True)}&scope=playlist-modify-public playlist-modify-private user-read-private user-read-email')

  _, bardo_id = su.get_user_data(token)
  needs_rating = db.load_tracks_to_rate(bardo_id)
  if check_recs and len(needs_rating) > 0:
    rate_url = url_for('rate_recommendations')
    params = get_request_params(request.args)
    return redirect(f'{rate_url}?bardo-id={bardo_id}&{params}&redirect-uri={redirect_uri}')

  return response(token=token, bardo_id=bardo_id, **kwargs)

def get_request_params(request, exclude=''):
  params = ''
  for name, param in request.items():
    if name != exclude:
      params += f'{name}={param}&'
  return params
