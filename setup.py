"""A setuptools module for Pathvision

See:
https://packaging.python.org/en/latest/distributing.html
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
# with open(path.join(here, 'README.md'), encoding='utf-8') as f:
#     long_description = f.read()

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'requirements.txt')) as f:
    install_requirements = f.read().splitlines()

setup(
    name='pathvision',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='1.0.0',

    description='Explainable AI Method for Object Detection models',
    # long_description="todo: readme",
    # long_description_content_type='text/markdown',

    # The project's main homepage.
    url='https://github.com/toby-j/pathvision',

    # Author details
    author='Toby Johnson',
    author_email='tobyojohnson@proton.me',

    # Choose your license
    license='Apache 2.0',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Data Scientists/Developers',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],

    # What does your project relate to?
    keywords='Pathvision',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),
    #package_dir={'': '.'},
    #packages=[''],

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=install_requirements,


    # TODO: Upon production, we should filter out development and production dependencies and only include extra ones for development here

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[full,tf1]
    # $ pip install -e ".[full,tf1]"  (if using zsh)
    # extras_require={
    #     "full": ['tensorflow>=1.15'],
    #     "tf1": ['tensorflow>=1.15'],
    # }
)
