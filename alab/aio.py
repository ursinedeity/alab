#!/usr/bin/env python

# Copyright (C) 2015 University of Southern California and
#                          Nan Hua
# 
# Authors: Nan Hua
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
__author__  = "Nan Hua"
__credits__ = ["Nan Hua","Ke Gong","Harianto Tjong"]

__license__ = "GPL"
__version__ = "0.0.1"
__email__   = "nhua@usc.edu"

import numpy as np
import os.path
import subprocess
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

def loadstream(filename):
    """
    Convert a file location, return a file handle
    zipped file are automaticaly unzipped using stream
    """
    if not os.path.isfile(filename):
        raise IOError("File %s doesn't exist!\n" % (filename))
    if os.path.splitext(filename)[1] == '.gz':
        p = subprocess.Popen(["zcat", filename], stdout = subprocess.PIPE)
        f = StringIO(p.communicate()[0])
    elif os.path.splitext(filename)[1] == '.bz2':
        p = subprocess.Popen(["bzip2 -d", filename], stdout = subprocess.PIPE)
        f = StringIO(p.communicate()[0])
    else:
        f = open(filename,'r')
    return f


################################################################################end bedgraph class
#def loadSubcompartment(filename,genome_version='mm9',info=5):
    ##need improvement
    #from flab.constant import mm9
    #from collections import namedtuple
    #genomeInfo = mm9.Genome()
    ## +++++++++++++++

    #bedInfo    = np.genfromtxt(filename,dtype=str)
    #resolution = int(bedInfo[0][2]) - int(bedInfo[0][1])
    #binInfo    = genomeInfo.get_fix_bin_info(resolution)

    #state      = np.chararray(binInfo.totalBin,itemsize=2)
    #state[:]   = 'NA'

    #for line in bedInfo:
        #i = int(line[6])-1
        #state[i] = line[info-1]

    #stateID   = {'NA':0,'A1':1,'A2':2,'B1':3,'B2':4,'B3':5}
    #stateInfo = [state,stateID]
    #return namedtuple('stateInfo', 'state, stateID')._make(stateInfo)



if __name__=='__main__':
    pass

