from setuptools import find_packages, setup

setup(
  name='app',
  version='1.0.0',
  packages=find_packages(),
  zip_safe=False,
  package_data={
    'app': ['templates/*', 'static/*', 'utils/*'],
  },
  install_requires=[
    'flask',
    'sklearn',
    'numpy',
    'requests',
    'matplotlib'
  ],
)
