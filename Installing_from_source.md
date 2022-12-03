## Common Installation Issues
1. This package uses GDAL and g++, both of which can be tricky to set up correctly.
GDAL in particular will often break after installing a new program
If you receive error messages such as the following:

```
ImportError: ~/anaconda3/envs/RAiDER/lib/python3.7/site-packages/matplotlib/../../../libstdc++.so.6: version `CXXABI_1.3.9' not found (required by ~/anaconda3/envs/RAiDER/lib/python3.7/site-packages/matplotlib/ft2font.cpython-37m-x86_64-linux-gnu.so)
ImportError: libtiledb.so.1.6.0: cannot open shared object file: No such file or directory
***cmake: ~/anaconda3/envs/RAiDER/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.20' not found (required by cmake)***
```

try running the following commands within your RAiDER conda environment:
```
conda update --force-reinstall libstdcxx-ng
conda update --force-reinstall gdal libgdal
```

2. This package requires both C++ and C headers, and the system headers are used for some C libraries. If running on a Mac computer, and "python setup.py build" results in a message stating that some system library header file is missing, try the following steps, and accept the various licenses and step through the installation process. Try re-running the build step after each update:

 ```
 xcode-select --install
 open /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg
 ```

## Testing your installation
To test the installation was successfull you can run the following tests:
```
py.test test/
raiderDelay.py -h
```

### To enable automatic CircleCI Tests from a pull requests

You will need to make sure that CircleCI is an authorized OAuth application from Github. Simply sign in [here](https://circleci.com/vcs-authorize/) using your github account.
