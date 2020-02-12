from setuptools import setup

def readme():
    with open('README.md', encoding='utf-8') as f:
        README = f.read()
    return README


setup(
      name='MVMO',
      version='1.0.6',
      description='Python package for heuristic optimization',
      long_description=readme(),
      long_description_content_type="text/markdown",
      url='https://github.com/dgusain1/MVMO',
      author='Digvijay Gusain',
      author_email='d.gusain@tudelft.nl',
      license='MIT',
      classifiers=[
              'License :: OSI Approved :: MIT License',
              'Programming Language :: Python :: 3',
              'Programming Language :: Python :: 3.6'],
      packages=['MVMO'],
      include_package_data=True,
      install_requires=['numpy', 'pandas', 'tqdm']     
      )