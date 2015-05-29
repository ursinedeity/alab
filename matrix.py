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

  #def _removeZeroEntry(self,maskGiven = 0):
    #if maskGiven == 0:
      #self._getZeroEntry()
    #self.matrix = np.delete(self.matrix,self.mask,0)
    #self.matrix = np.delete(self.matrix,self.mask,1)
    
  #def _expandZeroEntry(self):
    #for i in range(len(self.mask)):
      #self.mask[i] -= i
    #self.matrix = np.insert(self.matrix,self.mask,0,axis=0)
    #self.matrix = np.insert(self.matrix,self.mask,0,axis=1)
  #===================================================
  def krnorm(self,mask = None,**kwargs):
    from alab.krnorm import bnewt
    self._getMask(mask)
    x = bnewt(self.matrix,mask=self.mask,check=0,**kwargs)*100
    self.matrix *= x 
    self.matrix *= x.T
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
      self.matrix *= rowsum 
      self.matrix *= rowsum.T
      
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
    plotmatrix(figurename,np.log(self.matrix),**kwargs)
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
      
def plotmatrix(figurename,matrix,format='png',title=None,**kwargs):
  """Plot a 2D array with a colorbar.
  
  Parameters
  ----------
  
  matrix : a 2d numpy array
      A 2d array to plot
  cmap : matplotlib color map
      Color map used in matrix, e.g cm.Reds, cm.bwr
  clip_min : float, optional
      The lower clipping value. If an element of a matrix is <clip_min, it is
      plotted as clip_min.
  clip_max : float, optional
      The upper clipping value.
  label : str, optional
      Colorbar label
  ticklabels1 : list, optional
      Custom tick labels for the first dimension of the matrix.
  ticklabels2 : list, optional
      Custom tick labels for the second dimension of the matrix.
  """
  import matplotlib.pyplot as plt
  from matplotlib import cm
  
  clip_min = kwargs.pop('clip_min', -np.inf)
  clip_max = kwargs.pop('clip_max', np.inf)
  cmap     = kwargs.pop('cmap',cm.Reds)
  fig  = plt.figure()
  if 'ticklabels1' in kwargs:
    plt.yticks(range(matrix.shape[0]))
    plt.gca().set_yticklabels(kwargs.pop('ticklabels1'))
    
  if 'ticklabels2' in kwargs:
    plt.xticks(range(matrix.shape[1]))
    plt.gca().set_xticklabels(kwargs.pop('ticklabels2'))
    
  cax  = plt.imshow(np.clip(matrix, a_min=clip_min, a_max=clip_max),
                    interpolation='nearest',
                    cmap=cmap,
                    **kwargs)
  if title != None:
    plt.title(title)
    
  if 'label' not in kwargs:
    plt.colorbar()
  else:
    plt.colorbar().set_label(kwargs['label'])
    
  plt.show()

  if format == 'png':
    fig.savefig(figurename,dpi=600)
  elif format == 'pdf':
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(figurename)
    pp.savefig(fig,dpi=600)
    pp.close()
  
  plt.close(fig)
#===================================================
