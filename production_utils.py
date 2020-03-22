from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from train_utils import TrainUtil, print_line
import sample_generators as s
import spotify_utils as su
from sklearn import preprocessing


### Load labeled tracks
print('---Loading tracks DB---')
tracks = []
with open('tracks.txt') as f:
  for line in f:
    track = line.strip()
    tracks.append(track)
print_line()

# Train production classifiers
# These classifiers were picked through experimentation
def load_prod_classifiers():
  TEST_SIZE = 0.25
  t = TrainUtil(s.gen_pos_and_neg, 0.25, 6, True)
  # SVC
  svc_name = 'svc'
  svc = SVC(random_state=0)
  svc = t.train(svc_name, svc, True)
  # KNN
  knn_name = 'knn'
  knn = KNeighborsClassifier()
  parameters = [{
    'n_neighbors': list(range(1, 11)),
    'p': list(range(1, 6)),
    'weights': ['uniform', 'distance'],
  }]
  knn = t.train_cv(knn_name, knn, parameters, True)
  # GBDT
  gbdt_name = 'gbdt'
  gbdt = GradientBoostingClassifier()
  parameters = [{
    'n_estimators': [16, 32, 64, 100, 150, 200],
    'learning_rate': [0.0001, 0.001, 0.01, 0.025, 0.05,  0.1, 0.25, 0.5],
  }]
  gbdt = t.train_cv(gbdt_name, gbdt, parameters, True, True)

  return {
    svc_name: svc,
    knn_name: knn,
    gbdt_name: gbdt,
  }

# The general idea is to make a {limi} playlist per classifier
# On top, two other lists will include the average of the classifiers
# and random
def generate_recommendations(token, genres, classifiers, limit):
  playlists = {}
  for name in classifiers:
    playlists[name] = []
  playlists['random'] = []
  playlists['avg'] = []
  
  go_on = True
  while go_on:
    # We get 100 recommendations
    print_line()
    recommendations = su.get_recommendations(token, genres)
    for recommendation in recommendations:
      go_on = False
      if recommendation['id'] not in tracks:
        # Track is not labeled
        features = su.get_track_features(token, recommendation)
        # Standardize features for these classifiers
        scaler = t.get_scaler()
        features_scaled = scaler.transform([features])
        # Get predictions from all classifiers
        predictions_sum = 0.0
        for name, clf in classifiers.items():
          prediction = clf.predict(features_scaled)
          predictions_sum += prediction
          print(f'  {name} prediction: {prediction}')
          # We limit the number of tracks in the playlist
          # And we only care about positives
          if len(playlists[name]) < limit and prediction == 1:
            playlists[name].append(recommendation['id'])
      
        # Special cases
        print(f'  random prediction: 1')
        if len(playlists['random']) < limit:
          playlists['random'].append(recommendation['id'])
        prediction = predictions_sum / len(classifiers)
        print(f'  avg prediction: {prediction}')
        if len(playlists['avg']) < limit and prediction >= 0.5:
          playlists['avg'].append(recommendation['id'])
        #if prediction == 1:
        #  playlists['total'].append(recommendation['id'])
      else:
        print(f'{recommendation["name"]} already labeled')
  
      # Check if playlists are full
      for name, plst in playlists.items():
        if len(plst) < limit:
          go_on = True
          break
      if not go_on:
        break

  return playlists
