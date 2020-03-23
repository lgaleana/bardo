from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from train_utils import TrainUtil, print_line
import sample_generators as s
import spotify_utils as su
from sklearn import preprocessing


print(__name__)

### Load labeled tracks
print('---Loading tracks DB---')
tracks = []
with open('./tracks.txt') as f:
  for line in f:
    track = line.strip()
    tracks.append(track)
print_line()

# Train production classifiers
# These classifiers were picked through experimentation
TEST_SIZE = 0.25
t = TrainUtil(s.gen_very_pos_and_neg, 0.25, 6, True)
def load_prod_classifiers():
  # Linear SVC
  linear_svc_name = 'linear_svc'
  linear_svc = LinearSVC(dual=False)
  parameters = [{
    'C': [0.1, 1, 10, 100, 1000],
    'class_weight': [{1: w} for w in list(range(1, 11))],
  }]
#  linear_svc = t.train(linear_svc_name, linear_svc)
#  linear_svc = t.train(linear_svc_name, linear_svc, True)
#  linear_svc = t.train_cv(linear_svc_name, linear_svc, parameters, False, True)
  linear_svc = t.train_cv(linear_svc_name, linear_svc, parameters, True, True)

  # SVC
  svc_name = 'svc'
  svc = SVC(random_state=0)
  parameters = [{
    'kernel': ['rbf'],
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],
    'class_weight': [{1: w} for w in list(range(1, 11))],
  },
  {
    'kernel': ['sigmoid'],
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],
    'class_weight': [{1: w} for w in list(range(1, 11))],
  },
  {
    'kernel': ['polynomial'],
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],
    'degree': list(range(2, 7)),
    'class_weight': [{1: w} for w in list(range(1, 11))],
  }]
  svc = t.train(svc_name, svc, True)
#  svc = t.train_cv(svc_name, svc, parameters, True, True)

  # KNN
  knn_name = 'knn'
  knn = KNeighborsClassifier()
  parameters = [{
    'n_neighbors': list(range(1, 11)),
    'p': list(range(1, 6)),
    'weights': ['uniform', 'distance'],
  }]
#  knn = t.train(knn_name, knn, True)
#  knn = t.train_cv(knn_name, knn, parameters, True, True)

  # GBDT
  gbdt_name = 'gbdt'
  gbdt = GradientBoostingClassifier()
  parameters = [{
    'n_estimators': [16, 32, 64, 100, 150, 200],
    'learning_rate': [0.0001, 0.001, 0.01, 0.025, 0.05,  0.1, 0.25, 0.5],
  }]
#  gbdt = t.train(gbdt_name, gbdt)
#  gbdt = t.train(gbdt_name, gbdt, True)
#  gbdt = t.train_cv(gbdt_name, gbdt, parameters, False, True)
  gbdt = t.train_cv(gbdt_name, gbdt, parameters, True, True)
  print('---Finished training---')

  return {
    linear_svc_name: linear_svc,
    svc_name: svc,
#    knn_name: knn,
    gbdt_name: gbdt,
  }

# The general idea is to make a {limi} playlist per classifier
# On top, two other lists will include the average of the classifiers
# and random
def generate_recommendations(token, genres, classifiers, limit, standardize):
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
        # Get predictions from all classifiers
        predictions_sum = 0.0
        for name, clf in classifiers.items():
          # Standardize features
          if name in standardize:
            scaler = t.get_scaler()
            features_scaled = scaler.transform([features])
            prediction = clf.predict(features_scaled)
          else:
            prediction = clf.predict([features])
          predictions_sum += prediction
          print(f'  {name} prediction: {prediction}')
          # We limit the number of tracks in the playlist
          # And we only care about positives
          if len(playlists[name]) < limit and recommendation['id'] not in playlists[name] and prediction == 1:
            playlists[name].append(recommendation['id'])
      
        # Special cases
        print(f'  random prediction: 1')
        if len(playlists['random']) < limit and recommendation['id'] not in playlists['random']:
          playlists['random'].append(recommendation['id'])
        prediction = predictions_sum / len(classifiers)
        print(f'  avg prediction: {prediction}')
        if len(playlists['avg']) < limit and recommendation['id'] not in playlists['avg'] and prediction >= 0.5:
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
