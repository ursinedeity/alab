#!/usr/bin/env python

__author__ = 'N.H.'

import numpy as np
import os.path
import re
import matplotlib
matplotlib.use('Agg')

class contactmatrix(object):
  idxdtype = np.dtype([('chrom','S5'),('start',int),('end',int)])
  def __init__(self,filename,genome=None,resolution=None):
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
        if 'genome' in h5f.keys() and 'resolution' in h5f.keys():
          import cPickle
          self.genome     = cPickle.loads(h5f['genome'].value)
          self.resolution = cPickle.loads(h5f['resolution'].value)
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
    #----------------end filename
    if isinstance(genome,str) and isinstance(resolution,int):
      import alab.utils
      genomedb    = alab.utils.genome(genome,usechr=['#','X'])
      bininfo     = genomedb.bininfo(resolution)
      self.genome = genome
      self.resolution = resolution
      self.buildindex(bininfo.chromList,bininfo.startList,bininfo.endList)
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

  #========================================================normalization methods
  def krnorm(self,mask = None,**kwargs):
    """using krnorm balacing the matrix (overwriting the matrix!)
          
          mask is a 1-D vector with the same length as the matrix where 1s specify the row/column to be ignored
          if no mask is given, row/column with rowsum==0 will be automatically detected and ignored
        
          when large_mem is set to 1, matrix product is calculated using small chunks, 
          but this will slowdown the process a little bit.     
    """
    from alab.norm import bnewt
    self._getMask(mask)
    x = bnewt(self.matrix,mask=self.mask,check=0,**kwargs)*100
    self.matrix *= x 
    self.matrix *= x.T

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
      
  def icenorm(self,mask = None):
    from alab.numutils import ultracorrectSymmetricWithVector
    if mask is None:
      self._getMask(mask)
    else:
      self.matrix[mask,:]=0
      self.matrix[:,mask]=0
    
    self.matrix = ultracorrectSymmetricWithVector(self.matrix)
  #-----------------------------------------------------------------------------
  def removeDiagonal(self):
    np.fill_diagonal(self.matrix,0)
  
  def removePoorRegions(self, cutoff=1):
    """Removes "cutoff" percent of bins with least counts

    Parameters
    ----------
      cutoff : int, 0<cutoff<100
        Percent of lowest-counts bins to be removed
    """
    rowsum   = self.rowsum()
    mask     = np.flatnonzero(rowsum < np.percentile(rowsum[rowsum > 0],cutoff))
    self.matrix[mask,:] = 0
    self.matrix[:,mask] = 0
    
    
  def scale(self, cellaverage = 1):
    """
      Scale matrix so that average of cells is the given value. 
      By default, the rowsum will be the number of rows/columns
    """
    rowsum = self.rowsum()
    totalsum = rowsum.sum()
    self.matrix = self.matrix / totalsum * (cellaverage * (len(rowsum)-len(self.mask)) * (len(rowsum)-len(self.mask)))
  
  def range(self,chrom):
    """
      return the index range for a give chromsome
    """
    rangeList = np.flatnonzero(self.idx['chrom'] == chrom)
    return (rangeList[0],rangeList[-1])
  
  def makeIntraMatrix(self,chrom):
    """substract a chromsome matrix given a chromsome name
    chrom : str, chromosome name e.g 'chr1'
    """
    rstart,rend = self.range(chrom)
    submatrix   = contactmatrix(rend - rstart + 1)
    submatrix.matrix = self.matrix[rstart:rend+1,rstart:rend+1]
    submatrix.idx    = np.core.records.fromrecords(self.idx[rstart:rend+1],dtype=self.idxdtype)
    return submatrix
  #==============================================================plotting methods
  def plot(self,figurename,**kwargs):
    plotmatrix(figurename,np.log(self.matrix),**kwargs)
  
  def plotZeroCount(self,figurename,**kwargs):
    zeroCount = []
    for i in range(len(self.matrix)):
      zeros = len(np.flatnonzero(self.matrix[i] == 0))
      if zeros != len(self.matrix):
        zeroCount.append(zeros)
    #--endfor
    histogram(figurename,
              zeroCount,
              int(len(self.matrix)/100),
              xlab = '# of Zeros', ylab = 'Frequency',
              **kwargs)
    
    
  def plotSum(self,figurename,**kwargs):
    rowsum = self.rowsum()
    histogram(figurename,
              rowsum[rowsum > 0],
              int(len(self.matrix)/100),
              xlab = 'Row sums', ylab = 'Frequency',
              **kwargs)
    
  #==============================================================save method
  def save(self, filename, mod = 'hdf5', precision=3):
    if mod == 'npz':
      np.savez_compressed(filename,matrix=self.matrix,idx=self.idx)
    elif mod == 'hdf5':
      import h5py
      h5f = h5py.File(filename, 'w')
      h5f.create_dataset('matrix', data=self.matrix, compression = 'gzip', compression_opts=9)
      h5f.create_dataset('idx', data=self.idx, compression = 'gzip', compression_opts=9)
      try:
        self.genome
        self.resolution
      except NameError:
        pass
      else:
        h5f.create_dataset('genome',data = cPickle.dumps(self.genome))
        h5f.create_dataset('resolution',data = cPickle.dumps(self.resolution))
      h5f.close()
#------------------------------------------------------------------------------------------

def histogram(figurename, x, binNum, xlab=None, ylab=None, title=None, **kwargs):
  """Plot a frequency histogram with a given array x
  
  Parameters
  ----------
  x  : a 1-D vector
    Raw data
  binNum : int
    Number of bins to draw
  xlab : string, optional
    label for x axis
  ylab : string, optional
    label for y axis
  title : string, optional
    title of the figure
    
  """
  import matplotlib.pyplot as plt
  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.hist(x,binNum,**kwargs)
  if xlab != None:
    ax.set_xlabel(xlab)
  if ylab != None:
    ax.set_ylabel(ylab)
  if title != None:
    ax.title(title)
  
  plt.show()
  fig.savefig(figurename,dpi=600)
  plt.close(fig)
  
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
#--------------------------------------------------------------------
def compareMatrix(m1,m2,figurename = 'comparison.png',**kwargs):
  """compare 2 matrixes, output correlation coefficient
  
  Parameters
  ----------
    m1,m2 : contactmatrix instances
      must be the same dimensions
    figurename : str
      filename for columnwise pearson corr histogram, set None to escape this step
  """
  if not (isinstance(m1,contactmatrix) and isinstance(m2,contactmatrix)):
    raise TypeError, "Invalid argument type, 2 contactmatrixes are required"
  if len(m1) != len(m2):
    raise TypeError, "Invalid argument, dimensions of matrixes must meet"
  from scipy.stats import spearmanr,pearsonr
  
  flat1 = m1.matrix.flatten()
  flat2 = m2.matrix.flatten()
  nonzeros = (flat1 > 0) * (flat2 > 0)
  flat1 = flat1[nonzeros]
  flat2 = flat2[nonzeros]
  print 'pearsonr:'
  print pearsonr(flat1,flat2)
  print 'spearmanr:'
  print spearmanr(flat1,flat2)
  del flat1
  del flat2
  if not (figurename is None):
    corr = []
    for i in range(len(m1)):
      r = pearsonr(m1.matrix[i],m2.matrix[i])
      #print r
      if not np.isnan(r[0]):
        corr.append(r[0])
      
    histogram(figurename,
              corr,
              100,
              xlab = 'Correlation Coefficient', ylab = 'Frequency',
              **kwargs)
#----------------------------------------------------------------------
def loadh5dict(filename):
  import cPickle
  import h5py
  h5f    = h5py.File(filename,'r')
  genome           = cPickle.loads(h5f['genome'].value)
  resolution       = cPickle.loads(h5f['resolution'].value)
  #genomeIdxToLabel = cPickle.loads(h5f['genomeIdxToLabel'].value)
  binNumber        = cPickle.loads(h5f['binNumber'].value)
  newMatrix = contactmatrix(binNumber,genome,resolution)
  newMatrix.matrix[:] = h5f['heatmap'][:]
  return newMatrix