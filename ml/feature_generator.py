import bardo.utils.spotify_utils as su
import numpy as np

def get_audio_and_analysis_features(token, track):
  return (track, get_audio_features(token, [track])[0],
    get_analysis_features(token, track))

def get_audio_features(token, tracks):
  audio_features = su.get_tracks_features(token, tracks)

  useful_features = []
  for i, track in enumerate(tracks):
    features = audio_features[i]
    useful_features.append([
      track['popularity'],
      features['duration_ms'],
      features['acousticness'],
      features['danceability'],
      features['energy'],
      features['instrumentalness'],
      features['liveness'],
      features['loudness'],
      features['speechiness'],
      features['valence'],
    ])

  return useful_features

def get_analysis_features(token, track):
  analysis = su.get_track_analysis(token, track)

  # Track features
  useful_features = [analysis['track'].get('num_samples', -1)]
  # Bars
  bars = list(map(
    lambda bar: [f for f in bar.values()],
    filter(lambda bar: bar['confidence'] != -1, analysis['bars'])))
  useful_features.append(len(bars))
  useful_features += describe(bars)
  # Beats
  beats = list(map(
    lambda beat: [f for f in beat.values()],
    filter(lambda beat: beat['confidence'] != -1, analysis['beats'])))
  useful_features.append(len(beats))
  useful_features += describe(beats)
  # Tatums
  tatums = list(map(
    lambda tatum: [f for f in tatum.values()],
    filter(lambda tatum: tatum['confidence'] != -1, analysis['tatums'])))
  useful_features.append(len(tatums))
  useful_features += describe(tatums)
  # Sections
  sections = list(map(
    lambda s: [f for n, f in s.items() if n != 'key' and n!= 'key_confidence'],
    filter(lambda section: section['confidence'] != -1, analysis['sections'])))
  useful_features.append(len(sections))
  # Segments
  segments = []
  for segment in analysis['segments']:
    if segment['confidence'] != -1:
      segment_chunk = []
      for name, feature in segment.items():
        if name == 'pitches' or name == 'timbre':
          segment_chunk.extend(feature)
        elif name != 'loudness_end':
          segment_chunk.append(feature)
      segments.append(segment_chunk)
  useful_features.append(len(segments))

  return {
    'analysis': useful_features,
    'sections': sections,
    'segments': segments,
  }

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
  data = users_data.get(bardo_id, [])
  users = {'lsgaleana@gmail.com': 1, 'sheaney@gmail.com': 2}
  return [
    #users.get(bardo_id, 3),
    len(data),
    len(list(filter(lambda track: track['stars'] == 1, data))),
    len(list(filter(lambda track: track['stars'] == 2, data))),
    len(list(filter(lambda track: track['stars'] == 3, data))),
    len(list(filter(lambda track: track['stars'] == 4, data))),
    len(list(filter(lambda track: track['stars'] == 5, data))),
  ]

def describe(vec):
  des =  [
    np.mean(vec, axis=0),
    np.std(vec, axis=0),
    np.percentile(vec, 0, axis=0),
    np.percentile(vec, 25, axis=0),
    np.percentile(vec, 50, axis=0),
    np.percentile(vec, 75, axis=0),
    np.percentile(vec, 100, axis=0),
  ]
  return [f for d in des for f in d]

def pad_components(feature_map, p=None, seg_n=None):
  print('Padding components')
  if p is not None and seg_n is None:
    sec_lens = [
      len(r['sections']) * 10 for rs in feature_map.values() for r in rs.values()]
    max_ = int(np.percentile(sec_lens, p))
  elif p is not None: 
    seg_lens = [
      len(r['segments']) for rs in feature_map.values() for r in rs.values()]
    seg_step = int(np.percentile(seg_lens, p) / seg_n)
    max_ = seg_n * seg_step

  for rows in feature_map.values():
    for row in rows.values():
      if p is not None and seg_n is None:
        row['sections'] = pad_section(row['sections'], max_)
        row['segments'] = describe(row['segments'])
      elif p is not None:
        row['sections'] = describe(row['sections'])
        row['segments'] = pad_segment(row['segments'], seg_step, max_)
      else:
        row['sections'] = describe(row['sections'])
        row['segments'] = describe(row['segments'])

def pad_segment(segments, step, max_):
  SEG_LEN = 30
  topad = []
  size = len(segments)
  if size <= max_:
    for i in range(0, size, step):
      topad.extend(np.mean(segments[i:i + step], axis=0))
    pad = int((max_ / step * SEG_LEN - len(topad)) / 2)
    return np.pad(topad, (pad, pad), mode='constant').tolist()
  else:
    offset = int((size - max_) / 2)
    for i in range(offset, max_ + offset, step):
      topad.extend(np.mean(segments[i:i + step], axis=0))
    return topad

def pad_section(sections, max_):
  sf = [f for s in sections for f in s]
  size = len(sf)
  if size <= max_:
    pad = int((max_ - size) / 2)
    return np.pad(sf, (pad, pad), mode='constant').tolist()
  else:
    offset = int((size - max_) / 2)
    return sf[offset:max_ + offset]

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
