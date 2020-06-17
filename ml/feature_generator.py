import bardo.utils.spotify_utils as su
import numpy as np

def get_audio_features(token, tracks):
  audio_features = su.get_tracks_features(token, tracks)

  useful_features = []
  for i, track in enumerate(tracks):
    features = audio_features[i]
    useful_features.append([
      track['popularity'],
      features['duration_ms'],
      features['time_signature'],
      features['acousticness'],
      features['danceability'],
      features['energy'],
      features['instrumentalness'],
      features['liveness'],
      features['mode'],
      features['loudness'],
      features['speechiness'],
      features['valence'],
      features['tempo'],
    ])

  return useful_features

def get_analysis_features(token, track):
  analysis = su.get_track_analysis(token, track)

  useful_features = []
  # Track features
  track = analysis['track']
  useful_features = useful_features + [
    track.get('num_samples', -1),
    track.get('offset_seconds', -1),
    track.get('end_of_fade_in', -1),
    track.get('start_of_fade_out', -1),
  ]
  # Bars
  bars = analysis['bars']
  durations = list(map(
    lambda bar: bar['duration'],
    filter(lambda bar: 'confidence' in bar and bar['confidence'] != -1, bars),
  ))
  useful_features = useful_features + [
    len(bars),
    np.mean(durations),
    np.std(durations),
    np.var(durations),
  ]
  # Beats
  beats = analysis['beats']
  durations = list(map(
    lambda beat: beat['duration'],
    filter(lambda beat: 'confidence' in beat and beat['confidence'] != -1, beats),
  ))
  useful_features = useful_features + [
    len(beats),
    np.mean(durations),
    np.std(durations),
    np.var(durations),
  ]
  # Sections 
  sections = analysis['sections']
  durations = []
  loudness = []
  tempos = []
  modes = []
  ts = []
  for section in sections:
    durations.append(section['duration'])
    if 'loudness' in section:
      loudness.append(section['loudness'])
    if 'tempo' in section:
      tempos.append(section['tempo'])
    if 'mode' in section and section['mode'] != -1:
      modes.append(section['mode'])
    if 'time_signature' in section and section['time_signature'] != -1:
      ts.append(section['time_signature'])
  useful_features = useful_features + [
    len(sections),
    np.mean(durations),
    np.std(durations),
    np.var(durations),
    np.mean(loudness),
    np.std(loudness),
    np.var(loudness),
    np.mean(tempos),
    np.std(tempos),
    np.var(tempos),
    np.mean(modes),
    np.std(modes),
    np.var(modes),
    np.mean(ts),
    np.std(ts),
    np.var(ts),
  ]
  # Segments
  segments = analysis['segments']
  durations = []
  ls = []
  lmt = []
  lm = []
  pitches = []
  timbre = []
  for segment in segments:
    durations.append(segment['duration'])
    if 'loudness_start' in segment:
      ls.append(segment['loudness_start'])
    if 'loudness_max_time' in segment:
      lmt.append(segment['loudness_max_time'])
    if 'loudness_max' in segment:
      lm.append(segment['loudness_max'])
    pitches.append(segment['pitches'])
    timbre.append(segment['timbre'])
  useful_features = useful_features + [
    len(segments),
    np.mean(durations),
    np.std(durations),
    np.var(durations),
    np.mean(ls),
    np.std(ls),
    np.var(ls),
    np.mean(lmt),
    np.std(lmt),
    np.var(lmt),
    np.mean(lm),
    np.std(lm),
    np.var(lm),
  ]
  useful_features = useful_features + np.mean(pitches, axis=0).tolist() + np.mean(timbre, axis=0).tolist()
# Tatums
  tatums = analysis['tatums']
  durations = list(map(
    lambda tatum: tatum['duration'], 
    filter(lambda tatum: 'confidence' in tatum and tatum['confidence'] != -1, tatums),
  ))
  useful_features = useful_features + [
    len(tatums),
    np.mean(durations),
    np.std(durations),
    np.var(durations),
  ]

  return useful_features

def get_group_features(bardo_id, track, data):
  others_count = 0
  others_stars = 0
  for other_id, other_tracks in data.items():
    if bardo_id != other_id:
      for other_track in other_tracks:
        if track['id'] == other_track['id']:
          others_count += 1
          others_stars += other_track['stars']
          break

  return [
    others_count,
    others_stars,
  ]

def get_user_features(
  bardo_id,
  users_data,
):
  data = users_data[bardo_id]
  users = {'lsgaleana@gmail.com': 1, 'sheaney@gmail.com': 2}
  return [
    users.get(bardo_id, 3),
    #len(data),
    #len(list(filter(lambda track: track['stars'] == 1, data))),
    #len(list(filter(lambda track: track['stars'] == 2, data))),
    #len(list(filter(lambda track: track['stars'] == 3, data))),
    #len(list(filter(lambda track: track['stars'] == 4, data))),
    #len(list(filter(lambda track: track['stars'] == 5, data))),
  ]

def get_user_track_features(vectors, analysis):
  return [
    np.square(np.linalg.norm(vectors[0] - analysis)),
    np.square(np.linalg.norm(vectors[1] - analysis)),
    np.square(np.dot(vectors[0], analysis) / (np.linalg.norm(vectors[0]) * np.linalg.norm(analysis))),
    np.square(np.dot(vectors[1], analysis) / (np.linalg.norm(vectors[1]) * np.linalg.norm(analysis))),
  ]

def load_user_vectors(token, data):
  print('Generating user vectors')
  very_pos = filter(lambda track: track['stars'] == 5, data)
  pos = filter(lambda track: track['stars'] == 4, data)

  tracks = {}
  very_features = []
  for track in very_pos:
    analysis = get_analysis_features(token, track)
    tracks[track['id']] = analysis
    very_features.append(analysis)
  pos_features = []
  for track in pos:
    analysis = get_analysis_features(token, track)
    tracks[track['id']] = analysis
    pos_features.append(analysis)

  return (np.mean(very_features, axis=0), np.mean(pos_features, axis=0), tracks)
