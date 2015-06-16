#!/usr/bin/env python

__author__ = 'N.H.'

import numpy as np
cimport numpy as np 
cimport cython   

#==============================from mirnylib numutils=====================

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def ultracorrectSymmetricWithVector(x,v = None,M=None,diag = -1, 
                                    tolerance=1e-5):
  """Main method for correcting DS and SS read data. Possibly excludes diagonal.
  By default does iterative correction, but can perform an M-time correction"""
  if M == None:
    M = 599
  totalBias = np.ones(len(x),np.float32)    
  if v == None: v = np.zeros(len(x),np.float32)  #single-sided reads    
  #x = np.array(x,np.double,order = 'C')
  cdef np.ndarray[np.float32_t, ndim = 2] _x = x
  cdef np.ndarray[np.float32_t, ndim = 1] s 
  v = np.array(v,np.float32,order = "C")        
  cdef int i , j, N
  N = len(x)       
  for iternum in xrange(M):         
    s0 = np.sum(_x,axis = 1)         
    mask = [s0 == 0]            
    v[mask] = 0   #no SS reads if there are no DS reads here        
    nv = v / (totalBias * (totalBias[mask==False]).mean())
    s = s0 + nv
    for dd in xrange(diag + 1):   #excluding the diagonal 
      if dd == 0:
        s -= np.diagonal(_x)
      else:
        dia = np.array(np.diagonal(_x,dd))                
        s[dd:] = s[dd:] -  dia
        s[:len(s)-dd] = s[:len(s)-dd] - dia 
    s = s / np.mean(s[s0!=0])        
    s[s0==0] = 1
    s -= 1
    s *= 0.8
    s += 1   
    totalBias *= s
         
    for i in range(N):
      for j in range(N):
        _x[i,j] = _x[i,j] / (s[i] * s[j])
       
    if M == 599:
      if np.abs(s-1).max() < tolerance:
        #print "IC used {0} iterations".format(iternum+1)
        break

                         
  #corr = totalBias[s0!=0].mean()  #mean correction factor
  #x  = x * corr * corr #renormalizing everything
  #totalBias /= corr
  return _x
