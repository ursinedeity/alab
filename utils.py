#!/usr/bin/env python

__author__ = 'N.H.'
import os
import re
import math
import scipy
import warnings
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
#==========================================from mirny's lab source codes
def PCA(A, numPCs=6, verbose=False):
    """performs PCA analysis, and returns 6 best principal components
    result[0] is the first PC, etc"""
    #A = np.array(A, float)
    if np.sum(np.sum(A, axis=0) == 0) > 0 :
        warnings.warn("Columns with zero sum detected. Use zeroPCA instead")
    M = (A - np.mean(A.T, axis=1)).T
    covM = np.dot(M, M.T)
    [latent, coeff] = scipy.sparse.linalg.eigsh(covM, numPCs)
    if verbose:
        print "Eigenvalues are:", latent
    return (np.transpose(coeff[:, ::-1]), latent[::-1])


def EIG(A, numPCs=3):
    """Performs mean-centered engenvector expansion
    result[0] is the first EV, etc.;
    by default returns 3 EV
    """
    #A = np.array(A, float)
    if np.sum(np.sum(A, axis=0) == 0) > 0 :
        warnings.warn("Columns with zero sum detected. Use zeroEIG instead")
    M = (A - np.mean(A))  # subtract the mean (along columns)
    if isSymmetric(A):
        [latent, coeff] = scipy.sparse.linalg.eigsh(M, numPCs)
    else:
        [latent, coeff] = scipy.sparse.linalg.eigs(M, numPCs)
    alatent = np.argsort(np.abs(latent))
    print "eigenvalues are:", latent[alatent]
    coeff = coeff[:, alatent]
    return (np.transpose(coeff[:, ::-1]), latent[alatent][::-1])


def zeroPCA(data, numPCs=3, verbose=False):
    """
    PCA which takes into account bins with zero counts
    """
    nonzeroMask = np.sum(data, axis=0) > 0
    data = data[nonzeroMask]
    data = data[:, nonzeroMask]
    PCs = PCA(data, numPCs, verbose)
    PCNew = [np.zeros(len(nonzeroMask), dtype=float) for _ in PCs[0]]
    for i in range(len(PCs[0])):
        PCNew[i][nonzeroMask] = PCs[0][i]
    return PCNew, PCs[1]


def zeroEIG(data, numPCs=3):
    """
    Eigenvector expansion which takes into account bins with zero counts
    """
    nonzeroMask = np.sum(data, axis=0) > 0
    data = data[nonzeroMask]
    data = data[:, nonzeroMask]
    PCs = EIG(data, numPCs)
    PCNew = [np.zeros(len(nonzeroMask), dtype=float) for _ in PCs[0]]
    for i in range(len(PCs[0])):
        PCNew[i][nonzeroMask] = PCs[0][i]
    return PCNew, PCs[1]
  
