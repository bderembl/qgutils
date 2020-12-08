#!/usr/bin/env python

import numpy as np
import sys


# array handling


def reshape3d(dh,N2,f0=1, **kwargs):

  '''
  Convert all arrays to 3d arrays
  '''

  nd = N2.ndim
  si = N2.shape
  nl = si[0]
  si_z = len(dh)
  if nl != si_z - 1:
    print("N2 should be the size of dh - 1")
    sys.exit(1)

  psi = kwargs.get('psi', np.zeros(si_z))

  if np.isscalar(f0):
    f0 = np.array([f0]).reshape(1,1)

  if nd == 1:
    N2 = N2.reshape((nl,1,1))
    psi = psi.reshape((si_z,1,1))
    f0 = f0.reshape(1,1)
  elif nd == 2:
    N2 = N2.reshape((nl,si[1],1))
    psi = psi.reshape((si_z,si[1],1))
    f0 = f0.reshape(si[1],1)

  if 'psi' in kwargs:
    return N2,f0,psi
  elif f0 == 1:
    return N2
  else:
    return N2,f0


# Horizontal grid functions

def pad_bc(psi, bc='dirichlet'):
  """
  Pad field psi with Dirichlet or Neumann boundary conditions

  Parameters
  ----------

  psi : array [nz (,ny,nx)]
  bc   : 'dirichlet' or 'neumann'

  Returns
  -------

  psi_bc: array [nz (,ny+2,nx+2)]
  """

  nd = psi.ndim
  si = psi.shape
  
  if nd == 1:
    print("not handeling 1d arrays")
    sys.exit(1)
  elif nd == 2:
    psi = psi[None,:,:]

  # only pad horizontal dimensions
  psi = np.pad(psi,((0,0),(1,1),(1,1)),'constant')

  if (bc == 'dirichlet'): 
    psi[:,0,:]  = -psi[:,1,:]
    psi[:,-1,:] = -psi[:,-2,:]
    psi[:,:,0]  = -psi[:,:,1]
    psi[:,:,-1] = -psi[:,:,-2]
    
    # corners
    psi[:,0,0]   = -psi[:,0,1]   - psi[:,1,0]   - psi[:,1,1]
    psi[:,-1,0]  = -psi[:,-1,1]  - psi[:,-2,0]  - psi[:,-2,1]
    psi[:,0,-1]  = -psi[:,1,-1]  - psi[:,0,-2]  - psi[:,1,-2]
    psi[:,-1,-1] = -psi[:,-1,-2] - psi[:,-2,-2] - psi[:,-2,-1]

    # psi[:,0,:]  = 0.
    # psi[:,-1,:] = 0.
    # psi[:,:,0]  = 0.
    # psi[:,:,-1] = 0.
    
    # # corners
    # psi[:,0,0]   = 0.
    # psi[:,-1,0]  = 0.
    # psi[:,0,-1]  = 0.
    # psi[:,-1,-1] = 0.

  elif (bc == 'neumann'): 
    psi[:,0,:]  = psi[:,1,:]
    psi[:,-1,:] = psi[:,-2,:]
    psi[:,:,0]  = psi[:,:,1]
    psi[:,:,-1] = psi[:,:,-2]
    
    # corners
    psi[:,0,0]   = psi[:,1,1]
    psi[:,-1,0]  = psi[:,-2,1]
    psi[:,0,-1]  = psi[:,1,-2]
    psi[:,-1,-1] = psi[:,-2,-2]

  if nd == 2:
    return psi.squeeze()
  else:
    return psi


def coarsen(psi,n=1):
  """
  Coarsen input data by a factor 2 in each dimension

  Parameters
  ----------
  psi : ndarray [(nz ,) ny,nx]
    field to coarsen: can be 2d or 3d
  n   : int 
    number of coarsenings

  Returns
  -------
  psi_c: darray [(nz ,) ny/2^n,nx/2^n]
    coarsened array
  """

  nd = psi.ndim
  if nd == 2:
    psi = psi[None,:,:]

  for i in range(0,n):
    psi = 0.25*(psi[:,::2,::2] + psi[:,1::2,::2] + psi[:,::2,1::2] + psi[:,1::2,1::2])  

  if nd == 2:
    psi = np.squeeze(psi,0)

  return psi


def refine(psi, n=1, bc='dirichlet'):
  """
  Refine input data by linear prolongation

  Parameters
  ----------
  psi : narray [nz (,ny,nx)]
  n   : int 
    number of refinings
  bc : str
    type of boundary condition: 'dirichlet' or 'neumann'

  Returns
  -------
  psi_f: array [nz (,ny*2^n,nx*2^n)]
    refine array
  """

  nd = psi.ndim
  if nd == 2:
    psi = psi[None,:,:]

  for i in range(0,n):
    si = psi.shape
    psi_f = np.zeros((si[0], 2*si[1], 2*si[2]))

    psi = pad_bc(psi ,bc)
    psi_f[:,::2,::2]   = (9*psi[:,1:-1,1:-1] + 3*(psi[:,:-2,1:-1] + psi[:,1:-1,:-2]) + psi[:,:-2,:-2])/16
    psi_f[:,1::2,::2]  = (9*psi[:,1:-1,1:-1] + 3*(psi[:,2: ,1:-1] + psi[:,1:-1,:-2]) + psi[:,2: ,:-2])/16
    psi_f[:,::2,1::2]  = (9*psi[:,1:-1,1:-1] + 3*(psi[:,:-2,1:-1] + psi[:,1:-1, 2:]) + psi[:,:-2,2:])/16
    psi_f[:,1::2,1::2] = (9*psi[:,1:-1,1:-1] + 3*(psi[:,2: ,1:-1] + psi[:,1:-1, 2:]) + psi[:,2: ,2:])/16

    psi = psi_f

  if nd == 2:
    psi = np.squeeze(psi,0)

  return psi


def restriction(psi):
  nd = psi.ndim
  si = psi.shape


  psic = {}
  l = int(np.log2(si[1]))
  psic[l] = psi

  if nd == 2:
    psi = psi[None,:,:]

  while si[1] > 1:
    psi = coarsen(psi,1)
    si = psi.shape
    l = int(np.log2(si[1]))
    if nd == 2:
      psic[l] = np.squeeze(psi,0)
    else:
      psic[l] = psi

  return psic


def wavelet(psi, bc='dirichlet'):
  """
  Wavelet decomposition
  """

  nd = psi.ndim
  si = psi.shape

  psir = restriction(psi)

  depth = int(np.log2(si[1]))

  w = {}
  for l in range(depth-1,-1,-1):
    psip = refine(psir[l], 1, bc)
    w[l+1] = psir[l+1] - psip

  w[0] = psir[0]

  return w


def inverse_wavelet(w, bc='dirichlet'):
  """
  Wavelet reconsctruction
  """

  depth = len(w) - 1

  psi = w[0]

  for l in range(0,depth):
    psi = refine(psi, 1, bc) + w[l+1]

  return psi


def wavelet_lowpass(psi, filt_level, bc='dirichlet'):
  """
  Wavelet filter
  """
  nd = psi.ndim
  si = psi.shape
  depth = int(np.log2(si[1]))

  w = wavelet(psi,bc)
  for l in range(0,depth+1):
    if l > filt_level:
      w[l] = 0*w[l]

  psi = inverse_wavelet(w)
  return psi
