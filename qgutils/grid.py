#!/usr/bin/env python

import numpy as np
import sys


# array handling


def reshape3d(dh,N2, f0=1, **kwargs):

  '''
  Convert all arrays to 3d arrays

  ---> this function is a mess......

  '''

  nd = N2.ndim
  si = N2.shape
  nl = si[0]
  si_z = len(dh)
  if nl != si_z - 1:
    print("N2 should be the size of dh - 1")
    sys.exit(1)

  psi = kwargs.get('psi', np.zeros(si_z))
  returnf = 1

  if np.isscalar(f0):
    f0 = np.array([f0]).reshape(1,1)
    if f0 == 1:
      returnf = 0

  if nd == 1:
    N2 = N2.reshape((nl,1,1))
    f0 = f0.reshape(1,1)
  elif nd == 2:
    N2 = N2.reshape((nl,si[1],1))
    psi = psi.reshape((si_z,si[1],1))
    f0 = f0.reshape(si[1],1)

  if psi.ndim == 1:
    psi = psi.reshape((si_z,1,1))

  if 'psi' in kwargs:
    return N2,f0,psi
  elif returnf == 0:
    return N2
  else:
    return N2, f0


def field_type(psi):
  """
  Check if psi field is defined at cell center or cell node.

  We assume that if Nx (number of grid point in x) is even then
  the field is cell centered. 
  

  Parameters
  ----------

  psi : array [(nz), ny,nx]

  Returns
  -------

  'center' or 'node'
  """

  si = psi.shape
  N = si[-1]
  if N % 2 == 0:
    return 'center'
  else:
    return 'node'
    

# Horizontal grid functions
def interp_on_c(psi):
  """
  Interpolate field from cell corner to cell center.

  Parameters
  ----------

  psi : array [(nz), ny+1,nx+1]

  Returns
  -------

  psi: array [(nz), ny,nx]
  """
  
  return 0.25*(psi[...,1:,1:] + psi[...,:-1,:-1]
               + psi[...,1:,:-1]+ psi[...,:-1,1:])


def pad_bc(psi, bc='dirichlet'):
  """
  Pad field psi with Dirichlet (default), Neumann or periodic boundary conditions

  if psi is defined at cell centers, use 'dirichlet'
  if psi is defined at cell nodes, use 'dirichlet_face'


  with dirichlet: 

  Parameters
  ----------

  psi : array [ny,nx] or  [nz,ny,nx]
  bc   : 'dirichlet', 'dirichlet_face', 'neumann' or 'periodic'

  Returns
  -------

  psi_bc: array [ny+2,nx+2] or [nz,ny+2,nx+2]
  """

  nd = psi.ndim
  si = psi.shape
  
  if nd == 1:
    print("not handling 1d arrays")
    sys.exit(1)
  elif nd == 2:
    psi = psi[None,:,:]

  # only pad horizontal dimensions
  if bc:
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

  elif (bc == 'dirichlet_face'): 

    psi[:,0,:]  = 0.
    psi[:,-1,:] = 0.
    psi[:,:,0]  = 0.
    psi[:,:,-1] = 0.
    
    # corners
    psi[:,0,0]   = 0.
    psi[:,-1,0]  = 0.
    psi[:,0,-1]  = 0.
    psi[:,-1,-1] = 0.

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

  elif (bc == 'periodic'):
    psi[:,0,:]  = psi[:,-2,:]
    psi[:,-1,:] = psi[:,1,:]
    psi[:,:,0]  = psi[:,:,-2]
    psi[:,:,-1] = psi[:,:,1]

  elif (bc == None):
    # not much to do
    pass
  else:
    print("Boundary condition " + bc + " not implemented in pad_bc\n")

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
  si = psi.shape

  if nd == 2:
    psi = psi[None,:,:]

  for i in range(0,n):
    if si[-1] % 2 == 0: # centered field
      psi = 0.25*(psi[:,::2,::2] + psi[:,1::2,::2] + psi[:,::2,1::2] + psi[:,1::2,1::2])  

    else: # vertex field
      psic = (0.25*psi[:,2:-2:2,2:-2:2] +
              0.125*(psi[:,1:-3:2,2:-2:2] +
                     psi[:,3:-1:2,2:-2:2] +
                     psi[:,2:-2:2,1:-3:2] +
                     psi[:,2:-2:2,3:-1:2]) + 
              0.0625*(psi[:,1:-3:2,1:-3:2] + 
                      psi[:,1:-3:2,3:-1:2] +
                      psi[:,3:-1:2,1:-3:2] +
                      psi[:,3:-1:2,3:-1:2]))
      psic = pad_bc(psic,'dirichlet_face')

      # report original field boundary conditions
      psic[:,0,:]  = psi[:,0,::2] 
      psic[:,-1,:] = psi[:,-1,::2] 
      psic[:,:,0]  = psi[:,::2,0] 
      psic[:,:,-1] = psi[:,::2,-1] 
      
      psi = psic

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
  si = psi.shape

  if nd == 2:
    psi = psi[None,:,:]

  if si[-1] % 2 == 0: # centered field
    a = 0.5625; b = 0.1875; c = 0.0625
    for i in range(0,n):
      si = psi.shape
      psi_f = np.zeros((si[0], 2*si[1], 2*si[2]))

      psi = pad_bc(psi ,bc)
      psi_f[:,::2,::2]   = (a*psi[:,1:-1,1:-1] + b*(psi[:,:-2,1:-1] + psi[:,1:-1,:-2]) + c*psi[:,:-2,:-2])
      psi_f[:,1::2,::2]  = (a*psi[:,1:-1,1:-1] + b*(psi[:,2: ,1:-1] + psi[:,1:-1,:-2]) + c*psi[:,2: ,:-2])
      psi_f[:,::2,1::2]  = (a*psi[:,1:-1,1:-1] + b*(psi[:,:-2,1:-1] + psi[:,1:-1, 2:]) + c*psi[:,:-2,2:])
      psi_f[:,1::2,1::2] = (a*psi[:,1:-1,1:-1] + b*(psi[:,2: ,1:-1] + psi[:,1:-1, 2:]) + c*psi[:,2: ,2:])
      
      psi = psi_f

  else: # vertex field

    for i in range(0,n):
      si = psi.shape
      psi_f = np.zeros((si[0], 2*(si[1]-1)+1, 2*(si[2]-1)+1))

      # keep same BC as original field
      psi_f[:,::2, ::2] = psi[:,:,:]
      psi_f[:,1::2, 1::2] = 0.25*(psi[:,:-1,:-1] + psi[:,1:,:-1] + psi[:,:-1,1:] + psi[:,1:,1:])
      psi_f[:,::2, 1::2] = 0.5*(psi[:,:,:-1] + psi[:,:,1:])
      psi_f[:, 1::2, ::2] = 0.5*(psi[:,:-1,:] + psi[:,1:,:])
      
      psi = psi_f


  if nd == 2:
    psi = np.squeeze(psi,0)

  return psi

def smooth(psi, n=1, bc='dirichlet'):
  """
  Smooth data by coarsening and refining input data

  Parameters
  ----------
  psi : narray [nz (,ny,nx)]
  n   : int 
    number of coarsening and refining
  bc : str
    type of boundary condition: 'dirichlet' or 'neumann'

  Returns
  -------
  psi_f: array [nz (,ny*2^n,nx*2^n)]
    smoothed array
  """

  return refine(coarsen(psi,n),n, bc)


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


def wavelet_lowpass(psi, l_cut, Delta, bc='dirichlet'):
  """
  Wavelet filter


  Parameters
  ----------

  psi : array [(nz,)  ny,nx]
  l_cut : float or array [ny,nx]
  Delta   : grid step
  bc   : 'dirichlet' or 'neumann'

  Returns
  -------

  psi_filt: array [nz (,ny,nx)]

  """

  nd = psi.ndim
  si = psi.shape
  depth = int(np.log2(si[1]))

  L0 = si[1]*Delta

  if np.isscalar(l_cut):
    l_cut = l_cut*np.ones((si[1],si[1]))

  sig_lev = wavelet(l_cut,bc)
  sig_filt = restriction(l_cut)

  for l in range(depth,-1,-1):
    Delta = L0/2**l
    for j,i in np.ndindex((2**l,2**l)):
      ref_flag = 0
      if (l < depth):
        ref_flag = np.sum(sig_lev[l+1][2*j:2*j+2,2*i:2*i+2])
      if (ref_flag > 0):
        sig_lev[l][j,i] = 1
      else:
        if (sig_filt[l][j,i] > 2*Delta):
          sig_lev[l][j,i] = 0
        elif (sig_filt[l][j,i] <= 2*Delta and sig_filt[l][j,i] > Delta):
          sig_lev[l][j,i] = 1-(sig_filt[l][j,i]-Delta)/Delta
        else:
          sig_lev[l][j,i]= 1

  w = wavelet(psi,bc)

  for l in range(0,depth+1):
    w[l] = sig_lev[l]*w[l]

  psi_f = inverse_wavelet(w)
  return psi_f
