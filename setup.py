from setuptools import setup, find_packages

setup(
    name='map-boxes',
    version='1.0.1',
    description='Function to calculate mAP for set of detected boxes and annotated boxes.',
    author='Roman Solovyev, modified by Jeremi Wojcicki (ETA Compute)',
    author_email='jeremi@etacompute.com',
    url='git@github.com:Eta-Compute/MLToolbox.git',
    packages=find_packages(include=['map_boxes', 'map_boxes.*']),
    install_requires=[
		'pandas',
		'numpy',
    ]
)