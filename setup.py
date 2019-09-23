import os
from os.path import join as pjoin

from setuptools import setup, find_packages
PACKAGES = find_packages()

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 0
_version_micro = 1  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: Linux",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "NSDAccess: Access Natural Scenes Data."
# Long description will go up on the pypi page
long_description = """
This package provides a single class allowing the user to quickly and easily access the data from the Natural Scenes Dataset, see [the NSD project website](http://naturalscenesdataset.org).
It provides, in arbitraty volume or surface-based formats:
- one-line access to ROIs
- one-line access to arbitrary trial betas (of any provided type)
- one-line access to all behavioral output for arbitrary trials
- one-line access to all images in the dataset, and
- one-line access to the COCO annotations of all images in the dataset.

For this latter functionality, we use the [pycocotools](https://github.com/cocodataset/cocoapi), which need to be installed separately as they aren't pip-installable.
"""

NAME = "nsd_access"
MAINTAINER = "Tomas Knapen"
MAINTAINER_EMAIL = "tknapen@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/tknapen/nsd_access"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Tomas Knapen"
AUTHOR_EMAIL = "tknapen@gmail.com"
PLATFORMS = "Linux"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {} # {'nsd_access': [pjoin('data', '*')]}
REQUIRES = ["numpy", "sklearn", "scipy", "nibabel",
            "nilearn", "cifti", "h5py", "pyyaml", "pandas"]
DEP_LINKS = ['git+https://github.com/cocodataset/cocoapi/tree/master/PythonAPI']





opts = dict(name=NAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            version=VERSION,
            packages=PACKAGES,
            package_data=PACKAGE_DATA,
            install_requires=REQUIRES,
            requires=REQUIRES,
            dependency_links=DEP_LINKS)


if __name__ == '__main__':
    setup(**opts)