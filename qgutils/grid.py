#!/usr/bin/env python

import numpy as np
import sys

# Horizontal grid functions

def dirichlet_bc(psi):

  nd = psi.ndim
  si = psi.shape
  
  if nd == 1:
    print("not handeling 1d arrays")
    sys.exit(1)
  elif nd == 2: 
    psi = psi.reshape(1,si[0],si[1])

  # only pad horizontal dimensions
  psi = np.pad(psi,((0,0),(1,1),(1,1)))

  psi[:,0,:]  = -psi[:,1,:]
  psi[:,-1,:] = -psi[:,-2,:]
  psi[:,:,0]  = -psi[:,:,1]
  psi[:,:,-1] = -psi[:,:,-2]
  
  # corners
  psi[:,0,0]   = -psi[:,0,1]   - psi[:,1,0]   - psi[:,1,1]
  psi[:,-1,0]  = -psi[:,-1,1]  - psi[:,-2,0]  - psi[:,-2,1]
  psi[:,0,-1]  = -psi[:,1,-1]  - psi[:,0,-2]  - psi[:,1,-2]
  psi[:,-1,-1] = -psi[:,-1,-2] - psi[:,-2,-2] - psi[:,-2,-1]

  return psi.squeeze()


def neumann_bc(psi):

  nd = psi.ndim
  si = psi.shape
  
  if nd == 1:
    print("not handeling 1d arrays")
    sys.exit(1)
  elif nd == 2: 
    psi = psi.reshape(1,si[0],si[1])

  # only pad horizontal dimensions
  psi = np.pad(psi,((0,0),(1,1),(1,1)))

  psi[:,0,:]  = psi[:,1,:]
  psi[:,-1,:] = psi[:,-2,:]
  psi[:,:,0]  = psi[:,:,1]
  psi[:,:,-1] = psi[:,:,-2]
  
  # corners
  psi[0,0]   = psi[1,1]
  psi[-1,0]  = psi[-2,1]
  psi[0,-1]  = psi[1,-2]
  psi[-1,-1] = psi[-2,-2]

  return psi.squeeze()


def coarsen(psi,n):
  """
  Coarsen input data by a factor 2 in each dimension

  Parameters
  ----------

  psi : array [nz (,ny,nx)]
  n   : int number of coarsenings
  Returns
  -------

  psi_c: array [nz (,ny/2^n,nx/2^n)]
  """
  for i in range(0,n):
    psi = 0.25*(psi[:,::2,::2] + psi[:,1::2,::2] + psi[:,::2,1::2] + psi[:,1::2,1::2])  
  return psi
