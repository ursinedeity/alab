#!/usr/bin/env python

__author__ = 'N.H.'
import os
import re
import math
import numpy as np
from collections import namedtuple
from alab.io import loadstream

#===========================================================================
class genome(object):
  def __init__(self,genomeName,usechr=['#','X']):
    datafile = os.environ['GENOMES'] + '/' + genomeName + '.info'
    f = loadstream(datafile)
    self.info = np.genfromtxt(f,dtype=[('chrom','S5'),('length',int)])
    f.close() 
    removepattern = ''
    if not '#' in usechr: removepattern += '0-9'
    if not 'X' in usechr: removepattern += 'X'
    if not 'Y' in usechr: removepattern += 'Y'
    if not 'M' in usechr: removepattern += 'M'
    
    self.info = np.delete(self.info,np.nonzero([re.search('chr['+removepattern+']',c) for c in self.info['chrom']]))
    
  def bininfo(self,resolution):
    binSize    = [int(math.ceil(float(x)/resolution)) for x in self.info['length']]
    binStart   = [sum(binSize[:i]) for i in range(len(binSize))]
    
    chromList  = []
    binLabel   = []
    for i in range(len(self.info['chrom'])):
      chromList += [self.info['chrom'][i] for j in range(binSize[i])]
      binLabel     += [j for j in range(binSize[i])]
   
    startList  = [binLabel[j]*resolution for j in range(sum(binSize))]
    endList    = [binLabel[j]*resolution + resolution for j in range(sum(binSize))]
    
    binInfo    = [binSize,binStart,chromList,startList,endList]
    return namedtuple('binInfo', 'binSize, binStart, chromList, startList, endList')._make(binInfo)
  
  def getchrnum(self,chrom):
    findidx = np.flatnonzero(self.info['chrom']==chrom)
    
    if len(findidx) == 0:
      return -1
    else:
      return findidx[0]
  
  def getchrom(self,chromNum):
    assert isinstance(chromNum,int)
    return self.info['chrom'][chromNum]

def listadd(a,b):
  """
    Add 2 array/list that have different sizes
  """
  if len(a) < len(b):
    c = np.array(b).copy()
    c[:len(a)] += a
  else:
    c = np.array(a).copy()
    c[:len(b)] += b
  return c

def powerLawSmooth(matrix,w=3,s=3,p=3):
  csum = 0.0
  divider = 0.0
  for i in range(-w,w+1):
    for j in range(-w,w+1):
      decay = 1 / (abs(s*i) ** p + abs(s*j) ** p + 1.0)
      csum += matrix[w+i,w+j] * decay
      divider += decay
  
  return csum/divider