#!/usr/bin/env python

__author__ = 'N.H.'

import numpy as np
import os.path
import re
import matplotlib
matplotlib.use('Agg')

class contactmatrix(object):
  idxdtype = np.dtype([('chrom','S5'),('start',int),('end',int)])
  def __init__(self,filename,idx=None):
    if isinstance(filename,int):
      self.matrix=np.zeros((filename,filename),dtype = np.float32)
    elif isinstance(filename,str):
      if not os.path.isfile(filename):
	raise IOError,"File %s doesn't exist!\n" % (filename)
      if os.path.splitext(filename)[1] == '.npz':
	self.matrix = np.load(filename)['matrix']
	self.idx    = np.load(filename)['idx']
      elif os.path.splitext(filename)[1] == '.hdf5':
	import h5py
	h5f = h5py.File(filename,'r')
	self.matrix = h5f['matrix'][:]
	self.idx    = h5f['idx'][:]
	h5f.close()
      else:
	from alab.io import loadstream
	f    = loadstream(filename)
	s    = f.next()
	line = re.split('\t+|\s+',s.rstrip())
	n    = len(line) - 3
	idx  = []
	i    = 0
	idx.append(line[0:3])
	self.matrix = np.zeros((n,n),dtype = np.float32)
	self.matrix[i] = line[3:]
	for s in f:
	  i += 1
	  line = re.split('\t+|\s+',s.rstrip())
	  idx.append(line[0:3])
	  self.matrix[i] = line[3:]
	f.close()
	self.idx    = np.core.records.fromarrays(np.array(idx).transpose(),dtype=self.idxdtype)

  #==================================================
  def buildindex(self,chromlist,startlist,endlist):
    idxlist = np.column_stack([chromlist,startlist,endlist])
    self.idx = np.core.records.fromarrays(np.array(idxlist).transpose(),dtype=self.idxdtype)
  #--------------------------------------------------
  def __str__(self):
    return self.matrix.__str__()
  def __repr__(self):
    return self.matrix.__repr__()
  def __len__(self):
    return self.matrix.__len__()
  #def __del__(self):
    #try:
      #self.h5f.close()
    #except:
      #pass
  def rowsum(self):
    return self.matrix.sum(axis=1)
  def columnsum(self):
    return self.matrix.sum(axis=0)
  
  def _getZeroEntry(self):
    self.mask   = np.flatnonzero(self.rowsum() == 0)
  
  def _getMask(self,mask = None):
    if mask is None:
      self._getZeroEntry()
      return 0
    else:
      if isinstance(mask,np.ndarray):
	self.mask = mask
	return 1
      else:
	raise TypeError, "Invalid argument type, numpy.ndarray is required"

  def _removeZeroEntry(self,maskGiven = 0):
    if maskGiven == 0:
      self._getZeroEntry()
    self.matrix = np.delete(self.matrix,self.mask,0)
    self.matrix = np.delete(self.matrix,self.mask,1)
    
  def _expandZeroEntry(self):
    for i in range(len(self.mask)):
      self.mask[i] -= i
    self.matrix = np.insert(self.matrix,self.mask,0,axis=0)
    self.matrix = np.insert(self.matrix,self.mask,0,axis=1)
  #===================================================
  def krnorm(self,mask = None):
    from alab.krnorm import bnewt
    maskGiven = self._getMask(mask)
    self._removeZeroEntry(maskGiven)
    x = bnewt(self.matrix,check=0)*100
    self.matrix *= x * x.T
    self._expandZeroEntry()
  #====================================================
  def vcnorm(self,iterations=1,mask = None):
    self._getMask(mask)
    for i in range(iterations):
      print "\tIterations:",i+1
      rowsum   = self.rowsum()
      rowsum[self.mask] = 0
      totalsum = rowsum.sum()
      np.seterr(divide='ignore')
      rowsum   = 1/rowsum
      rowsum[self.mask] = 0
      self.matrix *= totalsum
      self.matrix *= rowsum * rowsum.T
      
  def icenorm(self, **kwargs):
    self.vcnorm(iterations=10, **kwargs)
  
  def removeDiagonal(self):
    np.fill_diagonal(self.matrix,0)
    
  def scale(self, cellaverage = 1):
    rowsum = self.rowsum()
    totalsum = rowsum.sum()
    self.matrix = self.matrix / totalsum * (cellaverage * (len(rowsum)-len(self.mask)) * (len(rowsum)-len(self.mask)))
  
  def range(self,chrom):
    rangeList = np.flatnonzero(self.idx['chrom'] == chrom)
    return (rangeList[0],rangeList[-1])
  
  def makeIntraMatrix(self,chrom):
    rstart,rend = self.range(chrom)
    submatrix   = contactmatrix(rend - rstart + 1)
    submatrix.matrix = self.matrix[rstart:rend+1,rstart:rend+1]
    submatrix.idx    = np.core.records.fromrecords(self.idx[rstart:rend+1],dtype=self.idxdtype)
    return submatrix
  #====================================================
  def plot(self,figurename,**kwargs):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    #if vmax is None: vmax = np.log(self.matrix.max())
    #
    fig  = plt.figure()
    cax  = plt.imshow(np.log(self.matrix), interpolation='nearest', cmap=cm.Reds, **kwargs)
    #cbar = fig.colorbar(cax, ticks=[0, m.matrix.mean()])
    plt.show()
    fig.savefig(figurename,dpi=300)
  #====================================================
  def save(self, filename, mod = 'hdf5', precision=3):
    if mod == 'npz':
      np.savez_compressed(filename,matrix=self.matrix,idx=self.idx)
    elif mod == 'hdf5':
      import h5py
      h5f = h5py.File(filename, 'w')
      h5f.create_dataset('matrix', data=self.matrix, compression = 'gzip', compression_opts=9)
      h5f.create_dataset('idx', data=self.idx, compression = 'gzip', compression_opts=9)
      h5f.close()
      