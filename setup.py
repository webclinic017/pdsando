#!/usr/bin/env python3

import sys
from distutils.core import setup
from pdsando.version import __version__

setup(
  name = 'pdsando',
  url = 'https://github.com/natsunlee/pdsando',
  version = __version__,
  author = 'Nathan Lee',
  author_email = 'lee.nathan.sh@gmail.com',
  install_requires = ['mplfinance', 'polygon-api-client', 'pdpipe', 'pandas', 'numpy', 'sklearn'],
  packages = [ 'pdsando', 'pdsando.api', 'pdsando.ta.datafeeds', 'pdsando.ta.pipeline', 'pdsando.ta.visualizations' ],
  license = 'Apache 2.0',
  long_description = 'Pandas Sando.'
)