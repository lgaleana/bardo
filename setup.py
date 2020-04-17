from setuptools import find_packages, setup

setup(
  name='bardo',
  version='0.0.1',
  packages=find_packages(),
  zip_safe=False,
  package_data={
    'bardo': ['templates/*', 'static/*', 'utils/*'],
  },
  install_requires=[
    'flask',
    'sklearn',
    'numpy',
    'requests',
    'matplotlib'
  ],
)
