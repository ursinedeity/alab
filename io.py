#!/usr/bin/env python

__author__ = 'N.H.'

import numpy as np
import os.path
import subprocess
from cStringIO import StringIO

def loadstream(filename):
    """
    Convert a file location, return a file handle
    zipped file are automaticaly unzipped using stream
    """
    if not os.path.isfile(filename):
        raise IOError,"File %s doesn't exist!\n" % (filename)
    if os.path.splitext(filename)[1] == '.gz':
        p = subprocess.Popen(["zcat", filename], stdout = subprocess.PIPE)
        f = StringIO(p.communicate()[0])
    elif os.path.splitext(filename)[1] == '.bz2':
        p = subprocess.Popen(["bzip2 -d", filename], stdout = subprocess.PIPE)
        f = StringIO(p.communicate()[0])
    else:
        f = open(filename,'r')
    return f


###############################################################################end bedgraph class
def loadSubcompartment(filename,genome_version='mm9',info=5):
    #need improvement
    from flab.constant import mm9
    from collections import namedtuple
    genomeInfo = mm9.Genome()
    # +++++++++++++++

    bedInfo    = np.genfromtxt(filename,dtype=str)
    resolution = int(bedInfo[0][2]) - int(bedInfo[0][1])
    binInfo    = genomeInfo.get_fix_bin_info(resolution)

    state      = np.chararray(binInfo.totalBin,itemsize=2)
    state[:]   = 'NA'

    for line in bedInfo:
        i = int(line[6])-1
        state[i] = line[info-1]

    stateID   = {'NA':0,'A1':1,'A2':2,'B1':3,'B2':4,'B3':5}
    stateInfo = [state,stateID]
    return namedtuple('stateInfo', 'state, stateID')._make(stateInfo)



if __name__=='__main__':
    pass

