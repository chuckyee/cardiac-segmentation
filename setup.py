from setuptools import setup
from setuptools import find_packages

setup(name='rvseg',
      version='0.1',
      description='Right ventricle cardiac MRI segmentation challenge',
      url='http://github.com/chuckyee/cardiac-segmentation',
      author='Chuck-Hou Yee',
      author_email='chuckyee@gmail.com',
      license='MIT',
      packages=['rvseg', 'rvseg.models'],
      zip_safe=False)
