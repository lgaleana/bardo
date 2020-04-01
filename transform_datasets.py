print('Generating intended dataset')
with open('datasets/tracks_all.txt') as ta:
  t = open('datasets/tracks.txt', 'w')
  d = open('datasets/dataset.txt', 'w')
  for line in ta:
    entries = line.strip().split('\t')
    star = float(entries[len(entries) - 1])
    if star != 3:
      track_ids = ','.join(entries[0:2])
      features = ','.join(entries[2:len(entries) - 1])
      if star != 3.5:
        t.write(f'{track_ids},{features},{star}\n')
        d.write(f'{features},{star}\n')
      else:
        t.write(f'{track_ids},{features},3.0\n')
        d.write(f'{features},3.0\n')       
  t.close()
  d.close()

print('Generating original dataset')
with open('datasets/tracks_all.txt') as ta:
  t = open('datasets/tracks_orig.txt', 'w')
  d = open('datasets/dataset_orig.txt', 'w')
  for line in ta:
    entries = line.strip().split('\t')
    star = float(entries[len(entries) - 1])
    if star != 3.5:
      track_ids = ','.join(entries[0:2]) 
      features = ','.join(entries[2:len(entries) - 1])
      t.write(f'{track_ids},{features},{star}\n')
      d.write(f'{features},{star}\n')
  t.close()
  d.close()

print('Generating full dataset')
with open('datasets/tracks_all.txt') as ta:
  t = open('datasets/tracks_full.txt', 'w')
  d = open('datasets/dataset_full.txt', 'w')
  for line in ta:
    entries = line.strip().split('\t')
    star = float(entries[len(entries) - 1])
    track_ids = ','.join(entries[0:2])
    features = ','.join(entries[2:len(entries) - 1])
    if star != 3.5:
      t.write(f'{track_ids},{features},{star}\n')
      d.write(f'{features},{star}\n')
    else:
      t.write(f'{track_ids},{features},3.0\n')
      d.write(f'{features},3.0\n')       
  t.close()
  d.close()

print('Generating mixed dataset')
with open('datasets/tracks_all.txt') as ta:
  t = open('datasets/tracks_mixed.txt', 'w')
  d = open('datasets/dataset_mixed.txt', 'w')
  for line in ta:
    entries = line.strip().split('\t')
    star = float(entries[len(entries) - 1])
    track_ids = ','.join(entries[0:2])
    features = ','.join(entries[2:len(entries) - 1])
    if star == 3.5:
      t.write(f'{track_ids},{features},4.0\n')
      d.write(f'{features},4.0\n')
    elif star == 3:
      t.write(f'{track_ids},{features},2.0\n')
      d.write(f'{features},2.0\n')       
    else:
      t.write(f'{track_ids},{features},{star}\n')
      d.write(f'{features},{star}\n')
  t.close()
  d.close()
