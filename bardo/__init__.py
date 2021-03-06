from flask import Flask, request, redirect, url_for, render_template
import bardo.utils.production_utils as pu
import bardo.utils.db_utils as db
import bardo.utils.spotify_utils as su
import ml.metrics as m
from datetime import datetime
import logging
from werkzeug.exceptions import InternalServerError

CLIENT_ID = '8de267b03c464274a3546bfe84496696'
VALID_SOURCES = {
  'Galeana': 'lsgaleana@gmail.com',
  'Heaney': 'sheaney@gmail.com',
}
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
      bardo_id=bardo_id,
      token=token,
      sources=VALID_SOURCES,
    )
  return validate_response(
    response,
    request,
    'playlist_selection',
    check_recs=False,
  )

@app.route('/generate-playlist')
def generate_playlist():
  source = request.args.get('source', 'lsgaleana@gmail.com')
  genres = request.args.getlist('genre')
  history = request.args.getlist('history')
  track = request.args.get('track')
  market = request.args.get('market', 'US')
  checked = request.args.get('checked', False)

  if not genres:
    genres = ['deep-house']

  def response(token, bardo_id, source, genres, history, track, market):
    return render_template(
      'generate-playlist.html',
      bardo_id=bardo_id,
      data={
        'token': token,
        'source': source,
        'genres': genres,
        'history': history,
        'track': track,
        'market': market,
      },
    )

  return validate_response(
    response,
    request,
    'generate_playlist',
    check_recs=(not checked),
    source=source,
    genres=genres,
    history=history,
    track=track,
    market=market,
  )

@app.route('/make-playlist', methods=['POST'])
def make_playlist():
  token = request.json.get('token')
  source = request.json.get('source')
  genres = request.json.get('genres')
  history = request.json.get('history')
  track = request.json.get('track')
  market = request.json.get('market')

  if not token and not source and not genres and not market:
    return INVALID_REQUEST

  sid, bardo_id = su.get_user_data(token)
  needs_rating = db.load_tracks_to_rate(bardo_id)
  if len(needs_rating) >= 30:
    return 'Please first rate previous recommendations.'

  clf_plsts, final_plst = pu.gen_recs(
    token,
    source,
    genres,
    history,
    track,
    market,
    PLAYLIST_LIMIT,
    TIME_LIMIT,
    bardo_id,
  )

  now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
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
  profile = db.load_profile_deduped(bardo_id).values()
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
    now = datetime.now().strftime("%Y-%m-%d")
    db.save_feedback(bardo_id, feedback, 'profile', f'{now}_{stars}')
    return '<meta name="viewport" content="width=device-width">Tracks saved.'
  else:
    return '<meta name="viewport" content="width=device-width">No tracks saved.'

@app.route('/rate-recommendations')
def rate_recommendations():
  def response(token, bardo_id):
    needs_rating = db.load_tracks_to_rate(bardo_id)
    save_url = url_for('save_ratings', bardo_id=bardo_id)
    save_url += '?' + get_request_params(request.args) + '&checked=true'
    return render_template(
      'rate-recommendations.html',
      needs_rating=list(needs_rating.items())[:10],
      size=len(needs_rating),
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
  if not redirect_uri:
    redirect_uri = 'main'

  now = datetime.now().strftime("%Y-%m-%d")
  db.save_feedback(bardo_id, db.process_feedback_input(
    bardo_id,
    request.form,
  ), 'feedback', now)

  redirect_url = url_for(redirect_uri)
  redirect_url += '?' + get_request_params(request.args, 'redirect-uri')
  return redirect(f'{redirect_url}')

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

@app.route('/faq')
def faq():
  return render_template('faq.html')

@app.route('/how-it-works')
def how_it_works():
  return render_template('how-it-works.html')

@app.route('/metrics')
def metrics():
  users_data = db.load_users_data('2020-04-25')
  metrics = m.calculate_metrics(users_data)
  return render_template('metrics.html', metrics=metrics)


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
  if check_recs:
    needs_rating = db.load_tracks_to_rate(bardo_id)
    if len(needs_rating) > 0:
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
