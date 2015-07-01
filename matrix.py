#!/usr/bin/env python

__author__ = 'N.H.'

import numpy as np
import os.path
import re
import h5py
import cPickle
import warnings
from alab.plots import plotxy, plotmatrix, histogram

class contactmatrix(object):
  idxdtype = np.dtype([('chrom','S5'),('start',int),('end',int)])
  def __init__(self,filename,genome=None,resolution=None):
    self._applyedMethods = {}
    if isinstance(filename,int):
      self.matrix=np.zeros((filename,filename),dtype = np.float32)
    elif isinstance(filename,str):
      if not os.path.isfile(filename):
        raise IOError,"File %s doesn't exist!\n" % (filename)
      if os.path.splitext(filename)[1] == '.npz':
        self.matrix = np.load(filename)['matrix']
        self.idx    = np.load(filename)['idx']
      elif os.path.splitext(filename)[1] == '.hdf5':
        h5f = h5py.File(filename,'r')
        self.matrix = h5f['matrix'][:]
        self.idx    = h5f['idx'][:]
        if 'applyedMethods' in h5f.keys():
          self._applyedMethods = cPickle.loads(h5f['applyedMethods'].value)
        
        if 'genome' in h5f.keys() and 'resolution' in h5f.keys():         
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
      self._buildindex(bininfo.chromList,bininfo.startList,bininfo.endList)
  #==================================================
  def _buildindex(self,chromlist,startlist,endlist):
    idxlist = np.column_stack([chromlist,startlist,endlist])
    self.idx = np.core.records.fromarrays(np.array(idxlist).transpose(),dtype=self.idxdtype)
  def buildindex(self,**kwargs):
    warnings.warn("buildindex is deprecated, specify genome and resolution instead of building index manually.", DeprecationWarning)
    self._buildindex(**kwargs)
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
  def applyed(self,method):
    if method in self._applyedMethods:
      return True
    else:
      return False
  
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
  def krnorm(self,mask = None,force=False,**kwargs):
    """using krnorm balacing the matrix (overwriting the matrix!)
       Parameters:
       -----------
       mask: list/array 
          mask is a 1-D vector with the same length as the matrix where 1s specify the row/column to be ignored
          or a 1-D vector specifing the indexes of row/column to be ignored
          if no mask is given, row/column with rowsum==0 will be automatically detected and ignored
       large_mem: bool
          when large_mem is set to 1, matrix product is calculated using small chunks, 
          but this will slowdown the process a little bit.     
    """
    if (not self.applyed('normalization')) or force:
      from alab.norm import bnewt
      self._getMask(mask)
      x = bnewt(self.matrix,mask=self.mask,check=0,**kwargs)*100
      self.matrix *= x 
      self.matrix *= x.T
      self._applyedMethods['normalization'] = 'krnorm'
    else:
      warnings.warn("Method %s was done before, use force = True to overwrite it." % (self._applyedMethods['normalization']))
  
  def vcnorm(self,iterations=1,mask = None,force=False):
    if (not self.applyed('normalization')) or force:
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
      self._applyedMethods['normalization'] = 'vcnorm'
    else:
      warnings.warn("Method %s was done before, use force = True to overwrite it." % (self._applyedMethods['normalization']))
      
  def icenorm(self,mask = None,force=False):
    if (not self.applyed('normalization')) or force:
      from alab.numutils import ultracorrectSymmetricWithVector
      if mask is None:
        self._getMask(mask)
      else:
        self.matrix[mask,:]=0
        self.matrix[:,mask]=0
      
      self.matrix = ultracorrectSymmetricWithVector(self.matrix)
      self._applyedMethods['normalization'] = 'icenorm'
    else:
      warnings.warn("Method %s was done before, use force = True to overwrite it." % (self._applyedMethods['normalization']))
  
  def diagnorm(self,countzero=False,norm=True,force=False):
    """ This function is to diagnol normalize matrix, 
 
        Parameters:
        -----------
        countzero: bool
          defines if we want to count zero values when calculate diagonal mean or not
        norm: bool
          if norm is set to False, this will only calculate the diagonal mean, sum, and count
          skipping the normalization step, the matrix will not be affected
    """
    if (not self.applyed('diagnorm')) or force:
      from alab.norm import diagnorm
      self.matrix, diagMean, diagSum, diagCount = diagnorm(self.matrix,countzero,norm)
      if norm:
        self._applyedMethods['diagnorm'] = [diagMean,diagSum,diagCount]
      return diagMean, diagSum, diagCount
    else:
      warnings.warn("Method diagnorm was done before, use force = True to overwrite it.")
      return self._applyedMethods['diagnorm'][0], self._applyedMethods['diagnorm'][1], self._applyedMethods['diagnorm'][2]
    
  #-----------------------------------------------------------------------------
  def removeDiagonal(self,force = False):
    if (not self.applyed('removeDiagonal')) or force:
      np.fill_diagonal(self.matrix,0)
      self._applyedMethods['removeDiagonal'] = True
    else:
      warnings.warn("Method removeDiagonal was done before, use force = True to overwrite it.")
      
  def removePoorRegions(self, cutoff=1, force = False):
    """Removes "cutoff" percent of bins with least counts

    Parameters
    ----------
      cutoff : int, 0<cutoff<100
        Percent of lowest-counts bins to be removed
    """
    if (not self.applyed('removePoorRegions')) or force:
      rowsum   = self.rowsum()
      mask     = np.flatnonzero(rowsum < np.percentile(rowsum[rowsum > 0],cutoff))
      self.matrix[mask,:] = 0
      self.matrix[:,mask] = 0
      self._applyedMethods['removePoorRegions'] = True
    else:
      warnings.warn("Method removePoorRegions was done before, use force = True to overwrite it.")
      
  def scale(self, cellaverage = 1):
    """
      Scale matrix so that average of cells is the given value. 
      By default, the rowsum will be the number of rows/columns
    """
    rowsum = self.rowsum()
    totalsum = rowsum.sum()
    try:
      self.mask
    except AttributeError:
      self._getMask()
    self.matrix = self.matrix / totalsum * (cellaverage * (len(rowsum)-len(self.mask)) * (len(rowsum)-len(self.mask)))
  
  def range(self,chrom):
    """
      return the index range for a give chromsome
    """
    rangeList = np.flatnonzero(self.idx['chrom'] == chrom)
    if len(rangeList)==0:
      raise ValueError, "%s is not found in the index" %(chrom)
    else:
      return (rangeList[0],rangeList[-1])
  
  def makeIntraMatrix(self,chrom):
    """substract a chromsome matrix given a chromsome name
    chrom : str, chromosome name e.g 'chr1'
    """
    if self.applyed('subMatrix'):
      warnings.warn("This is already a submatrix!")
    try:
      rstart,rend = self.range(chrom)
    except ValueError:
      raise ValueError, "%s is not found in the index. Possibly you are not using the genome wide matrix" %(chrom)
    submatrix   = contactmatrix(rend - rstart + 1)
    submatrix.matrix = self.matrix[rstart:rend+1,rstart:rend+1]
    submatrix.idx    = np.core.records.fromrecords(self.idx[rstart:rend+1],dtype=self.idxdtype)
    try:
      self.genome
      self.resolution
    except AttributeError:
      warnings.warn("No genome and resolution is specified, attributes are recommended for matrix.")
    else:
      submatrix.genome     = self.genome
      submatrix.resolution = self.resolution
    submatrix._applyedMethods['subMatrix'] = chrom
    return submatrix
  #==============================================================plotting methods
  def plot(self,figurename,log=True,**kwargs):
    """
    plot the matrix heat map
    Parameters:
    -----------
    figurename : str
    log: bool
      if True, plot the log scale of the matrix
      if False, plot the original matrix
    clip_max:
    clip_min:
      2 options that will clip the matrix to certain value
    cmap:
      color map of the matrix
    label:
      label of the figure
    """
    if log:
      plotmatrix(figurename,np.log(self.matrix),**kwargs)
    else:
      plotmatrix(figurename,self.matrix,**kwargs)
  
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
    
    
  def plotSum(self,figurename,outlier=False,line=None,**kwargs):
    """
    Print the rowsum frequency histogram
    
    Parameters:
    -----------
    figurename: string
      Name of the plot
    outlier: bool
      option to select plotting the outlier line, only functioning if 'line' parameter is set to None
    line: float/array/list
      draw vertical lines at a list of positions 
    """
    rowsum = self.rowsum()
    if line is None:
      if outlier:
        line = (np.percentile(rowsum,75) - np.percentile(rowsum,25))*1.5 + np.percentile(rowsum,75)
      
    histogram(figurename,
              rowsum[rowsum > 0],
              int(len(self.matrix)/100),
              xlab = 'Row sums', ylab = 'Frequency',
              line = line,
              **kwargs)
  def plotHiCscoreVSDistance(self,figurename=None,genome=None,resolution=None,background=False,**kwargs):
    """
      plot distal ratio for intra chromosomes
      Parameters:
      -----------
      genome: str
        if no genome info found along with the matrix, genome parameter must be specified
        hg/mm9..
    """
    import alab.utils
    try:
      self.genome
      self.resolution
    except AttributeError:
      warnings.warn("No genome and resolution is specified, attributes are recommended for matrix.")
      if (genome is None) or (resolution is None):
        raise ValueError, "No genome info is found! Genome and resolution parameter must be specified."
      else:
        self.genome = genome
        self.resolution = resolution
        
    genomedb = alab.utils.genome(self.genome,usechr=['#','X'])
    TotalSums   = []
    TotalCounts = []
    for chrom in genomedb.info['chrom']:
      tmpMatrix = self.makeIntraMatrix(chrom)
      sums, counts = tmpMatrix.diagnorm(norm=False)[1:3] #skip the division step
      TotalSums   = alab.utils.listadd(TotalSums,sums)
      TotalCounts = alab.utils.listadd(TotalCounts,counts)
    HiCscore = TotalSums/TotalCounts
    dist = np.empty(len(HiCscore))
    for i in range(len(HiCscore)):
      dist[i] = i*int(self.resolution)
    
    if background:
      intermask = self.idx['chrom'][:,None] != self.idx['chrom'][None,:]
      hline = self.matrix[intermask].mean()
    else:
      hline=None
    if figurename is None:
      return dist, HiCscore
    else:
      plotxy(figurename,np.log10(dist),np.log10(HiCscore),hline=np.log10(hline),**kwargs)
      
  #==============================================================save method
  def save(self, filename):
    """
      Save the matrix along with information in hdf5 file
    """
    if (filename[-5:] != '.hdf5'):
      filename += '.hdf5'
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('matrix', data=self.matrix, compression = 'gzip', compression_opts=9)
    h5f.create_dataset('idx', data=self.idx, compression = 'gzip', compression_opts=9)
    h5f.create_dataset('applyedMethods', data=cPickle.dumps(self._applyedMethods))
    try:
      self.genome
      self.resolution
    except AttributeError:
      warnings.warn("No genome and resolution is specified, attributes are recommended for matrix.")
    else:
      h5f.create_dataset('genome',data = cPickle.dumps(self.genome))
      h5f.create_dataset('resolution',data = cPickle.dumps(self.resolution))
    h5f.close()
#------------------------------------------------------------------------------------------


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
  h5f    = h5py.File(filename,'r')
  genome           = cPickle.loads(h5f['genome'].value)
  resolution       = cPickle.loads(h5f['resolution'].value)
  #genomeIdxToLabel = cPickle.loads(h5f['genomeIdxToLabel'].value)
  binNumber        = cPickle.loads(h5f['binNumber'].value)
  newMatrix = contactmatrix(binNumber,genome,resolution)
  newMatrix.matrix[:] = h5f['heatmap'][:]
  return newMatrix