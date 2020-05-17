def calculate_metrics(users_data):
  metrics = {}
  for user, data in users_data.items():
    profile, clf_predictions = data[0], data[1]
    
    clf_metrics = {}
    for clf, predictions in clf_predictions.items():
      n = 0
      stars = 0
      pr_neutral = 0
      pr_pos = 0
      for prediction in predictions:
        if prediction['id'] in profile:
          s = profile[prediction['id']]['stars']
          n += 1
          stars += s
          if s > 2:
            pr_neutral += 1
          if s > 3:
            pr_pos += 1

      if n > 0:
        stars = stars / n
        pr_neutral = pr_neutral / n
        pr_pos = pr_pos / n
        clf_metrics[clf] = (n, stars, pr_neutral, pr_pos)

    if len(clf_metrics) > 0:
      metrics[user] = clf_metrics

  return metrics
