from setuptools import find_packages, setup

setup(
  name='app',
  version='1.0.0',
  packages=find_packages(),
  zip_safe=False,
  package_data={
    'datasets': ['*.txt'],
    'app': ['templates/*', 'static/*', 'utils/*'],
  },
  #data_files=[
  #  ('datasets', ['datasets/dataset_all.txt', 'datasets/tracks.txt']),
  #],
  install_requires=[
    'flask',
    'sklearn',
    'numpy',
    'requests',
    'matplotlib'
  ],
)
