from flask import Flask, request, redirect, url_for, render_template
import production_utils as pu
import server_utils as su

app = Flask(__name__)

CLIENT_ID = '8de267b03c464274a3546bfe84496696'
PLAYLIST_LIMIT = 20

pu.load_prod_classifiers()

@app.route('/')
def main():
  return render_template(
    'index.html',
    playlist_url=url_for('generate_playlist'),
    rate_url=url_for('profile'),
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
    needs_rating = su.load_tracks_to_rate(bardo_id)
    if len(needs_rating) == 0:
      code = request.args.get('code')
      if code:
        su.make_playlist(
          code,
          url_for('spotify_auth', _external=True),
          PLAYLIST_LIMIT,
        )
        return render_template('generate-playlist.html')
      else:
        auth_uri = url_for('spotify_auth', _external=True)
        return redirect(
          f'https://accounts.spotify.com/authorize?client_id={CLIENT_ID}&response_type=code&redirect_uri={auth_uri}&scope=playlist-modify-public playlist-modify-private&show_dialog=true',
        )
    else:
      return redirect(
        f'{url_for("rate_recommendations")}?bardo-id={bardo_id}&redirect-uri=generate_playlist'
      )
  else:
    generate_url = url_for("generate_playlist").replace('/', '')
    return redirect(
      f'{url_for("identify")}?redirect-url={generate_url}'
    )

@app.route('/spotify-auth')
def spotify_auth():
  code = request.args.get('code')
  if code:
    return render_template('spotify-auth.html', code=code)
  else:
    return 'Invalid request'

@app.route('/profile')
def profile():
  bardo_id = request.args.get('bardo-id')
  if bardo_id:
    needs_rating = su.load_tracks_to_rate(bardo_id)
    if len(needs_rating) == 0:
      return render_template('profile.html')
    else:
      return redirect(
        f'{url_for("rate_recommendations")}?bardo-id={bardo_id}&redirect-uri=profile'
      )
  else:
    generate_url = url_for("profile").replace('/', '')
    return redirect(
      f'{url_for("identify")}?redirect-url={generate_url}'
    )

@app.route('/rate-recommendations')
def rate_recommendations():
  redirect_uri = request.args.get('redirect-uri')
  if not redirect_uri:
    redirect_uri = 'main'

  bardo_id = request.args.get('bardo-id')
  if bardo_id:
    needs_rating = su.load_tracks_to_rate(bardo_id)
    save_url = url_for('save_ratings')
    return render_template(
      'rate-recommendations.html',
      needs_rating=needs_rating,
      save_url=f'{save_url}?bardo-id={bardo_id}&redirect-uri={redirect_uri}',
    )
  else:
    rate_url = url_for("rate_recommendations").replace('/', '')
    return redirect(
      f'{url_for("identify")}?redirect-url={rate_url}'
    )

@app.route('/save-ratings', methods=['POST'])
def save_ratings():
  bardo_id = request.args.get('bardo-id')
  redirect_uri = request.args.get('redirect-uri')

  if bardo_id and redirect_uri:
    su.save_feedback(
      bardo_id,
      request.form.getlist('like'),
      request.form.getlist('no-like'),
      request.form.getlist('default'),
    )
    return redirect(f'{url_for(redirect_uri)}?bardo-id={bardo_id}')
  else:
    return 'Invalid request.'
