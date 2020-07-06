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
  useful_features = [
    analysis['track'].get('num_samples', 0.0),
    len(analysis['sections']),
  ]

  # Iterators
  bars = iter(analysis['bars'])
  beats = iter(analysis['beats'])
  tatums = iter(analysis['tatums'])
  segments = iter(analysis['segments'])
  bar = next(bars)
  beat = next(beats)
  tatum = next(tatums)
  segment = next(segments)
  # Normalize all pieces of data to the sections
  sections = []
  for section in analysis['sections']:
    end = section['start'] + section['duration']
    section_chunk = []
    for name, feature in section.items():
      if name != 'key' and '_confidence' not in name:
        if section['confidence'] != -1:
          section_chunk.append(feature)
        else:
          section_chunk.append(0.0)
    # Bars
    bar_chunks = []
    try:
      while bar['start'] < end:
        if bar['confidence'] != -1:
          bar_chunks.append([bar['duration'], bar['confidence']])
          bar = next(bars)
    except StopIteration:
      pass
    section_chunk.append(len(bar_chunks))
    section_chunk.extend(np.mean(bar_chunks, axis=0))
    # Beats
    beat_chunks = []
    try:
      while beat['start'] < end:
        if beat['confidence'] != -1:
          beat_chunks.append([beat['duration'], beat['confidence']])
          beat = next(beats)
    except StopIteration:
      pass
    section_chunk.append(len(beat_chunks))
    section_chunk.extend(np.mean(beat_chunks, axis=0))
    # Tatums
    tatum_chunks = []
    try:
      while tatum['start'] < end:
        if tatum['confidence'] != -1:
          tatum_chunks.append([tatum['duration'], tatum['confidence']])
          tatum = next(tatums)
    except StopIteration:
      pass
    section_chunk.append(len(tatum_chunks))
    section_chunk.extend(np.mean(tatum_chunks, axis=0))
    # Segments
    segment_chunks = []
    try:
      while segment['start'] < end:
        if segment['confidence'] != -1:
          seg = []
          for segname, segfeature in segment.items():
            if segname == 'duration' or segname == 'confidence' or segname == 'loudness_max_time' or segname == 'loudness_max':
              seg.append(segfeature)
          segment_chunks.append(seg)
          segment = next(segments)
    except StopIteration:
      pass
    section_chunk.append(len(segment_chunks))
    section_chunk.extend(np.mean(segment_chunks, axis=0))

    sections.append(section_chunk)

  # Segments
  segments = []
  for segment in analysis['segments']:
    if segment['confidence'] != -1:
      segment_chunk = []
      for name, feature in segment.items():
        if name == 'pitches' or name == 'timbre':
          segment_chunk.extend(feature)
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
  SEC_LEN = 21
  if p is not None and seg_n is None:
    sec_lens = [
      len(r['sections']) * SEC_LEN for rs in feature_map.values() for r in rs.values()]
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

def pad_section(sections, max_):
  sf = [f for s in sections for f in s]
  size = len(sf)
  if size <= max_:
    pad = int((max_ - size) / 2)
    return np.pad(
      sf,
      (pad, pad if (max_ - size) % 2 == 0 else pad + 1),
      mode='constant',
    ).tolist()
  else:
    offset = int((size - max_) / 2)
    return sf[offset:max_ + offset]

def pad_segment(segments, step, max_):
  SEG_LEN = 24
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
