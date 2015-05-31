#!/usr/bin/env python

__author__ = 'N.H.'

import numpy as np
import math
import time
import numutils

def chunking_dot(big_matrix, small_matrix, chunk_size=10000):
  # Make a copy if the array is not already contiguous
  small_matrix = np.ascontiguousarray(small_matrix)
  R = np.empty((big_matrix.shape[0], small_matrix.shape[1]))
  for i in range(0, R.shape[0], chunk_size):
    end = i + chunk_size
    R[i:end] = np.dot(big_matrix[i:end], small_matrix)
  return R

def bnewt(A, mask=[], tol = 1e-6, delta_lower = 0.1, delta_upper = 3, fl = 0, check = 1, largemem = 0, chunk_size = 10000):
  # BNEWT A balancing algorithm for symmetric matrices
  #
  # X = BNEWT(A) attempts to find a vector X such that
  # diag(X)*A*diag(X) is close to doubly stochastic. A must
  # be symmetric and nonnegative.
  #
  # X0: initial guess. TOL: error tolerance.
  # delta/Delta: how close/far balancing vectors can get
  # to/from the edge of the positive cone.
  # We use a relative measure on the size of elements.
  # FL: intermediate convergence statistics on/off.
  # RES: residual error, measured by norm(diag(x)*A*x - e).
  
  
  (n,m) = A.shape
  #np.seterr(divide='ignore')
  print 'Verifying Matrix\n'
  if (n != m):
    print 'Matrix must be symmetric to converge\n'
    return 'NaN'
  if (check):
    for i in range(0,n):
      for j in range(i,n):
        if (A[i][j] != A[j][i])or(A[i][j] < 0):
          print 'Matrix must be symmetric and nonnegative to converge\n'
          return 'NaN'
    print 'Check OK\n'
  else:
    print 'Check escaped\n'
  
  e        = np.ones((n,1))
  e[mask]  = 0
  #res      = np.empty((n,1))
  
  g        = 0.9
  etamax   = 0.1
  eta      = etamax
  stop_tol = tol*0.5
  x        = e #initial guess
  rt       = tol*tol
  if largemem:
    v      = x * chunking_dot(A,x,chunk_size=chunk_size)
  else:
    v      = x*np.dot(A,x)
  rk       = 1 - v
  rk[mask] = 0
  rho_km1  = np.dot(np.transpose(rk),rk)
  rout     = rho_km1
  rold     = rout
  
  MVP = 0 #matrix vector products
  i = 0
  
  while rout > rt:
    i = i+1
    k=0
    y=e
    innertol = max(eta*eta*rout,rt)
    
    while rho_km1 > innertol: #inner iteration by CG
      k = k+1
      if k==1:
        with np.errstate(invalid='ignore'):
          Z       = rk/v
        Z[mask] = 0
        p       = Z
        rho_km1 = np.dot(np.transpose(rk),Z)
      else:
        beta = rho_km1/rho_km2
        p    =  Z + beta*p
      
      #update search direction 
      if largemem:
        w   = x*chunking_dot(A,x*p,chunk_size=chunk_size) + v*p
      else:
        w   = x*np.dot(A,x*p) + v*p
      
      alpha = rho_km1/np.dot(np.transpose(p),w)
      ap = alpha*p
      
      #test distance to boundary of cone
      ynew = y + ap
      if min(np.delete(ynew,mask)) <= delta_lower:
        if delta_lower == 0:
          break
        else:
          ind = np.nonzero(ap < 0)
          gamma = min((delta_lower - y[ind])/ap[ind])
          y = y + gamma*ap
          break
        if max(ynew) >= delta_upper:
          ind = np.nonzero(ynew > delta_upper)
          gamma = min((delta_upper-y[ind])/ap[ind])
          y = y + gamma*ap
          break
      
      y       = ynew
      rk      = rk - alpha*w
      rho_km2 = rho_km1
      with np.errstate(invalid='ignore'):
        Z       = rk/v
      Z[mask] = 0
      rho_km1 = np.dot(np.transpose(rk),Z)
    #end inner iteration
    
    x        = x*y
    if largemem:
      v      = x * chunking_dot(A,x,chunk_size=chunk_size)
    else:
      v      = x*np.dot(A,x)
    rk       = 1-v
    rk[mask] = 0
    rho_km1  = np.dot(np.transpose(rk),rk)
    rout     = rho_km1
    MVP      = MVP + k + 1
    #print MVP,res
    #update inner iteration stopping criterion
    rat      = rout/rold
    rold     = rout
    res_norm = math.sqrt(rout)
    eta_o    = eta
    eta      = g*rat
    
    if g*eta_o*eta_o > 0.1:
      eta = max(eta,g*eta_o*eta_o)
    eta = max(min(eta,etamax),stop_tol/res_norm)
    
    if fl == 1:
      print '%3d %6d %.3f \n' % (i,k,r_norm)
      
    if MVP > 50000:
      break
  #end outer
  
  print 'Matrix vector products = %6d\n' % MVP
  #x = np.array(x)
  #x[mask] = 0
  return x

#==============================from mirnylib numutils=====================
ultracorrectSymmetricWithVector = \
  numutils.ultracorrectSymmetricWithVector  # @UndefinedVariable @IgnorePep8
#==========================================================================


#def krnorm(A,tol=1e-6):
  ## this is an implimentation of bnewt with non-symmetric matrix
  ## matrix must be square to convergence
  
  #(n,m) = A.shape
  #if (n != m):
    #print 'Matrix must be square to converge!\n'
    #return 'NaN'
  #Zero_upper = np.zeros((n,n))
  #Zero_lower = np.zeros((m,m))
  #S = np.concatenate((np.concatenate((Zero_upper,A),axis=1),np.concatenate((A.T,Zero_lower),axis=1)))
  ##S=np.concatenate((A.T,Zero_lower),axis=1)
  ##print S
  
  #x=bnewt(S)
  
  #print x
  #R = S * x * x.T
  #Rc = R[0:n:1,m::1]
  ##xc = x[m::1]
  #return Rc
  
if __name__=='__main__':
  
  A = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[100,200,300,400]])
  #A = np.array([[1,2,3,40],[2,5,6,70],[3,6,8,90],[40,70,90,100]])
  R = krnorm(A)
  print R