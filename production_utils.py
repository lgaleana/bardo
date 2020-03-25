from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from train_utils import TrainUtil, print_line
import sample_generators as s
import spotify_utils as su
from sklearn import preprocessing


### Training configs
# These classifiers were picked through experimentation
# They will be used in production
train_configs = [
  {
    'name': 'linear_svc',
    'clf': LinearSVC(dual=False),
    'generator':,
    'standardize': True,
    'cv': False,
    'parameters': [{
      'C': [0.1, 1, 10, 100, 1000],
      'class_weight': [{1: w} for w in list(range(1, 11))],
    }],
    'active': False,
  },
  {
    'name': 'svc',
    'clf': SVC(random_state=0),
    'standardize': True,
    'cv': False,
    'parameters': [{
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
    }],
    'active': True,
  },
  {
    'name': 'knn',
    'clf': KNeighborsClassifier(),
    'generator':,
    'standardize': True,
    'cv': True,
    'parameters': [{
      'n_neighbors': list(range(1, 11)),
      'p': list(range(1, 6)),
      'weights': ['uniform', 'distance'],
    }],
    'active': False,
  },
  {
    'name': 'gbdt',
    'clf': GradientBoostingClassifier(random_state=0),
    'generator':,
    'standardize': True,
    'cv': True,
    'parameters': [{
      'n_estimators': [16, 32, 64, 100, 150, 200],
      'learning_rate': [0.0001, 0.001, 0.01, 0.025, 0.05,  0.1, 0.25, 0.5],
    }],
    'active': True,
  },
]

### Load labeled tracks
# So that we don't recommend them
print('---Loading tracks DB---')
tracks = []
with open('./tracks.txt') as f:
  for line in f:
    track = line.strip()
    tracks.append(track)
print_line()

### Train production classifiers
TEST_SIZE = 0.25
CV = 6
def load_prod_classifiers():

  return {
#    linear_svc_name: linear_svc,
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
