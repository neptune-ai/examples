import os
import sys

from glob import glob
from subprocess import check_call

import pytest

test_files = glob('**/notebooks/*.ipynb', recursive=True)

excluded_files = []
if os.name == 'nt': # if OS is Windows
    # Excluding because unable to install Tensorflow on a Windows CI server with
    #
    # ERROR: Could not install packages due to an EnvironmentError: [WinError 5] Access is denied: 'c:\\hostedtoolcache\\windows\\python\\3.6.8\\x64\\lib\\site-packages\\~umpy\\.libs\\libopenblas.NOIJJG62EMASZI6NYURL6JBKM4EVBGM7.gfortran-win_amd64.dll'
    # Consider using the `--user` option or check the permissions.
    excluded_files.extend(glob('product-tours\\how-it-works\\tests\\*.py', recursive=True)) #TODO

 # if OS is Windows and Python is 3.8
if os.name == 'nt' and sys.version_info.major == 3 and sys.version_info.minor == 8:
    # Excluding because on a Windows CI server with Python 3.8, tkinter error occurs.
    #
    # Traceback (most recent call last):
    #
    #   File "c:\hostedtoolcache\windows\python\3.8.6\x64\lib\tkinter\__init__.py", line 4014, in __del__
    #
    #     self.tk.call('image', 'delete', self.name)
    #
    # RuntimeError: main thread is not in main loop
    excluded_files.append('integrations\\xgboost\\notebooks\\Neptune-XGBoost_upgraded_libs.py') #TODO


@pytest.mark.parametrize("filename", [f for f in test_files if f not in excluded_files])
def test_examples(filename):
    # run all the noteboks with ipython filename
    # check_call('ipython ' + filename, shell=True)
    pass