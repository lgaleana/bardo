from flask import Flask, request, redirect, url_for, render_template
import production_utils as pu
import server_utils as s
import spotify_utils as su
from datetime import datetime

app = Flask(__name__)

CLIENT_ID = '8de267b03c464274a3546bfe84496696'
PLAYLIST_LIMIT = 10
EXP_CONFIG = [
  'svc_cv_very_balanced',
  'gbdt_cv_very',
  'gbdt_very_high',
  'svc_cv_very',
]
post_auth = 'main'

pu.load_prod_classifiers()

@app.route('/')
def main():
  return render_template(
    'index.html',
    playlist_url=url_for('playlist_selection'),
    rate_url=url_for('profile'),
  )

@app.route('/playlist-selection')
def playlist_selection():
  bardo_id = request.args.get('bardo-id')
  token = request.args.get('token')
  bardo_param = ''
  if bardo_id:
    bardo_param = f'&bardo-id={bardo_id}'
  if token:
    generate_url = url_for('generate_playlist')
    exp_clfs = ','.join(EXP_CONFIG)
    return render_template(
      'playlist-selection.html',
      bardo_url=f'{generate_url}?token={token}&engine=bardo{bardo_param}',
      random_url=f'{generate_url}?&token={token}&engine=random{bardo_param}',
      experiment_url=f'{generate_url}?&token={token}&engine={exp_clfs}{bardo_param}',
    )
  else:
    global post_auth
    post_auth = 'playlist_selection'
    return redirect(f'https://accounts.spotify.com/authorize?client_id={CLIENT_ID}&response_type=code&redirect_uri={url_for("spotify_auth", _external=True)}&scope=playlist-modify-public playlist-modify-private')

@app.route('/generate-playlist')
def generate_playlist():
  bardo_id = request.args.get('bardo-id')
  token = request.args.get('token')
  engine = request.args.get('engine')
  if not engine:
    engine = 'bardo'

  genres = ['deep-house']
  now = datetime.now().strftime("%d-%m-%Y_%H-%M")
  exp_config = engine.split(',')

  if token:
    if bardo_id:
      needs_rating = s.load_tracks_to_rate(bardo_id)
      if len(needs_rating) == 0:
        s.make_playlist(token, genres, PLAYLIST_LIMIT, now, exp_config)
        return render_template('generate-playlist.html')
      else:
        return redirect(
          f'{url_for("rate_recommendations")}?bardo-id={bardo_id}&token={token}&engine={engine}&redirect-uri=generate_playlist'
        )
    else:
      return redirect(
        f'{url_for("identify")}?token={token}&engine={engine}&redirect-uri=generate_playlist'
      )
  else:
    return 'Invalid request.'

@app.route('/profile')
def profile():
  bardo_id = request.args.get('bardo-id')
  if bardo_id:
    needs_rating = s.load_tracks_to_rate(bardo_id)
    if len(needs_rating) == 0:
      token = request.args.get('token')
      if token:
        liked_url = url_for('tracks', bardo_id=bardo_id, label='liked')
        not_liked_url = url_for('tracks', bardo_id=bardo_id, label='not-liked')
        all_url = url_for('tracks', bardo_id=bardo_id, label='all')
        save_url = url_for('save_playlists', bardo_id=bardo_id)
        return render_template(
          'profile.html',
          liked_url=liked_url,
          not_liked_url=not_liked_url,
          all_url=all_url,
          save_url=f'{save_url}?token={token}',
        )
      else:
        global post_auth
        post_auth = 'profile'
        return redirect(f'https://accounts.spotify.com/authorize?client_id={CLIENT_ID}&response_type=code&redirect_uri={url_for("spotify_auth", _external=True)}&scope=playlist-modify-public playlist-modify-private')
    else:
      return redirect(
        f'{url_for("rate_recommendations")}?bardo-id={bardo_id}&redirect-uri=profile'
      )
  else:
    return redirect(
      f'{url_for("identify")}?redirect-uri=profile'
    )

@app.route('/tracks/<bardo_id>/<label>')
def tracks(bardo_id, label):
  profile = s.load_profile(bardo_id)
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
    feedback = s.process_plst_feedback(
      token,
      favorite_url,
      not_favorite_url,
    )
    if len(feedback) > 0:
      now = datetime.now().strftime("%d-%m-%Y")
      s.save_feedback(bardo_id, feedback, 'profile', now)
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
    redirect_url=url_for(redirect_uri),
    args=args,
  )

@app.route('/rate-recommendations')
def rate_recommendations():
  bardo_id = request.args.get('bardo-id')
  if bardo_id:
    needs_rating = s.load_tracks_to_rate(bardo_id)
    save_url = url_for('save_ratings', bardo_id=bardo_id)
    save_url += '?'
    for name, param in request.args.items():
      save_url += f'{name}={param}&'
    return render_template(
      'rate-recommendations.html',
      needs_rating=needs_rating,
      save_url=f'{save_url}',
    )
  else:
    return redirect(
      f'{url_for("identify")}?redirect-uri=rate_recommendations'
    )

@app.route('/save-ratings/<bardo_id>', methods=['POST'])
def save_ratings(bardo_id):
  redirect_uri = request.args.get('redirect-uri')

  now = datetime.now().strftime("%d-%m-%Y_%H-%M")
  s.save_feedback(bardo_id, s.process_feedback_input(
    s.load_tracks_to_rate(bardo_id),
    request.form,
  ), 'feedback', now)

  if redirect_uri:
    redirect_url = url_for(redirect_uri)
    redirect_url += '?'
    for name, param in request.args.items():
      if name != 'redirect-uri':
        redirect_url += f'{name}={param}&'
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
