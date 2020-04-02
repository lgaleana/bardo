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
      if star != 4:
        t.write(f'{track_ids}\t{features}\t{star}\n')
        d.write(f'{features}\t{star}\n')
      else:
        t.write(f'{track_ids}\t{features}\t3.0\n')
        d.write(f'{features}\t3.0\n')       
  t.close()
  d.close()
