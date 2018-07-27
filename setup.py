from setuptools import setup

__version__ = ['0.0']

setup(name='torchzoo',
      version='.'.join(__version__),
      description='Various Implementations of neural models defined in papers',
      url='http://gitlab.com/theSage21/torchzoo',
      author='Arjoonn Sharma',
      author_email='arjoonn.94@gmail.com',
      packages=['torchzoo'],
      install_requires=['pytorch'],
      keywords=['torchzoo', 'pytorch', 'paper implementation'],
      zip_safe=False)
