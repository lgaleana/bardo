from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from train_utils import TrainUtil
import sample_generators as s
import spotify_utils as su
from sklearn import preprocessing

### Train production classifiers
# These classifiers were picked through experimentation
TEST_SIZE = 0.25
t = TrainUtil(s.gen_very_pos_and_neg, TEST_SIZE)
# Linear SVC
linear_svc_name = 'linear_svc'
svc = LinearSVC(dual=False)
linear_svc = t.train(linear_svc_name, svc, True)
print('--------------------------------------------------------')
# SVC
svc_name = 'svc'
svc = SVC(random_state=0)
svc = t.train(svc_name, svc, True)
print('--------------------------------------------------------')
# GBDT
gbdt_name = 'gbdt'
gbdt = GradientBoostingClassifier()
parameters = [{
  'n_estimators': [16, 32, 64, 100, 150, 200],
  'learning_rate': [0.0001, 0.001, 0.01, 0.025, 0.05,  0.1, 0.25, 0.5],
}]
gbdt = t.train_cv(gbdt_name, gbdt, parameters, True)

### Load labeled tracks
print('---Loading tracks DB---')
tracks = []
with open('tracks.txt') as f:
  track = f.readline().strip()
  tracks.append(track)
print()

### Make playlists
# The general idea is to make a 10-song playlist per classifier
# On top, two other lists will include the average of the classifiers
# and the weighted average
PLAYLIST_LIMIT = 10
classifiers = {
  linear_svc_name: linear_svc,
  svc_name: svc,
  gbdt_name: gbdt,
}
playlists = {}
for name in classifiers:
  playlists[name] = []
playlists['avg'] = []
genres = ['deep-house']

token = su.request_token()
go_on = True
while go_on:
  # We get 100 recommendations
  recommendations = su.get_recommendations(token, genres)
  for recommendation in recommendations:
    go_on = False
    features = su.get_track_features(token, recommendation)
    # Standardize features for these classifiers
    scaler = t.get_scaler()
    features_scaled = scaler.transform([features])
    # Get predictions from all classifiers
    predictions_sum = 0.0
    for name, clf in classifiers.items():
      prediction = clf.predict(features_scaled)
      print(f'  {name} prediction: {prediction}')
      # We limit the number of tracks in the playlist
      if len(playlists[name]) < PLAYLIST_LIMIT:
        if recommendation['id'] not in tracks:
          # Track is not labeled
          if prediction == 1:
            playlists[name].append(recommendation['id'])
        else:
          print(f'  {recommendation["name"]} already labeled')
      predictions_sum += prediction

    # Special cases
    prediction = predictions_sum / len(classifiers)
    print(f'  avg prediction: {prediction}')
    if prediction >= 0.5:
      playlists['avg'].append(recommendation['id'])
    #if prediction == 1:
    #  playlists['total'].append(recommendation['id'])

    for name, plst in playlists.items():
      if len(plst) < PLAYLIST_LIMIT:
        # All playlists are full
        go_on = True
    if not go_on:
      break

# Save playlists
for name, plst in playlists.items():
  f_plst = open(f'playlists/{name}.txt', 'w')
  for recommendation in plst:
    f_plst.write(f'{recommendation}\n')
  f_plst.close()
