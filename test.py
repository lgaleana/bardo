import spotify_utils as s

token = s.request_token()

r = s.get_recommendations(token, ['deeep-list'])
prev_recs = set(map(lambda rec: rec['name'], r))
all_recs = prev_recs
for i in range(30):
  r = s.get_recommendations(token, ['deep-house'])
  recs = set(map(lambda rec: rec['name'], r))
  print(len(recs - prev_recs))
  print(len(recs - all_recs))
  prev_recs = recs
  all_recs.update(recs)
