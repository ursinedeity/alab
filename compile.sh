
cython -a numutils.pyx

gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
      -I/usr/usc/python/2.7.8/include/python2.7/ -o numutils.so numutils.c