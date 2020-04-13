#!/usr/bin/env python

import numpy as np
import scipy.linalg as la
#from scipy.sparse import diags
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import sys

from .grid import *

# PV related stuff

def reshape3d(dh,N2,f0, **kwargs):

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
  else:
    return N2,f0


def gamma_stretch(dh, N2, f0=1.0, wmode=False, squeeze=True) :
  # sqeeze is intended as an internal option
  '''
  '''

  N2,f0 = reshape3d(dh,N2,f0)
  nl,si_y,si_x = N2.shape

  if wmode:
    S = np.zeros((nl,nl,si_y,si_x))
  else:
    S = np.zeros((nl+1,nl+1,si_y,si_x))

  sloc = np.zeros((nl+1,nl+1))
  for j,i in np.ndindex((si_y,si_x)):
  
    dhi = 0.5*(dh[1:] + dh[:-1])
  
    diag_p1 = 1/(dh[:-1]*dhi*N2[:,j,i])
    diag_m1 = 1/(dh[1:]*dhi*N2[:,j,i])
  
    if wmode:
      diag0 = -diag_p1 - diag_m1    
      # switch diagonals on purpose for wmode
      #      diagonals = [diag0,diag_p1[1:],diag_m1[:-1]]
      np.fill_diagonal(S[:,:,j,i],f0[j,i]**2*diag0)
      np.fill_diagonal(S[1:,:,j,i],f0[j,i]**2*diag_p1[1:])
      np.fill_diagonal(S[:,1:,j,i],f0[j,i]**2*diag_m1[:-1])
    else:
      diag0 = -np.append(diag_p1,0.)
      diag0[1:] = diag0[1:] - diag_m1
      #      diagonals = [diag0,diag_m1,diag_p1]
      np.fill_diagonal(S[:,:,j,i],f0[j,i]**2*diag0)
      np.fill_diagonal(S[1:,:,j,i],f0[j,i]**2*diag_m1)
      np.fill_diagonal(S[:,1:,j,i],f0[j,i]**2*diag_p1)

  
  
#    Sloc = diags(diagonals, [0, -1, 1])
  
    # Sloc = f0[j,i]**2*Sloc
  
    # if not sparse:
    #   Sloc = Sloc.toarray()
    # else:
    #   print("sparse option not working yet (issue with multi dim arrays)")
  
    # S[:,:,j,i] = Sloc


  if squeeze:
    return S.squeeze()
  else:
    return S

def comp_modes(dh, N2, f0=1.0, eivec=False, wmode=False):
  '''compute eigenvalues (and eigenvectors) of the sturm-liouville
  equation 

       d  ( f^2  d     )     1
       -- ( ---  -- psi)  + ---- psi = 0
       dz ( N^2  dz    )    Rd^2

  for a given stratification

  The eigenvectors correspond to the matrices for the mode/layer
  conversion

  mod2lay[:,0] is the barotropic mode: should be 1..1
  mod2lay[:,i] is the ith baroclinic mode

  to convert from physical to modal
  u_mod = np.dot(lay2mod[:,:],u_lev)
  np.einsum('ijkl,jkl->ikl',lay2mod,u_lev)

  to go back to the physical space
  u_lev = np.dot(mod2lay[:,:],u_mod)

  the w_modes are related to the p_modes by
  w_modes = -1/N2 d p_modes/dz

  '''

  N2,f0 = reshape3d(dh,N2,f0)
  nl,si_y,si_x = N2.shape

  S = gamma_stretch(dh,N2,f0, squeeze=False)

  # put variables in right format
  Ht = np.sum(dh)
  dhi = 0.5*(dh[1:] + dh[:-1])
  dhcol = dh[:,None]
  dhicol = dhi[:,None]


  if wmode:
    Rd = np.zeros((nl,si_y,si_x))
    if eivec:
      mod2lay = np.zeros((nl,nl,si_y,si_x))
      lay2mod = np.zeros((nl,nl,si_y,si_x))
  else:
    Rd = np.zeros((nl+1,si_y,si_x))
    if eivec:
      mod2lay = np.zeros((nl+1,nl+1,si_y,si_x))
      lay2mod = np.zeros((nl+1,nl+1,si_y,si_x))

  for j,i in np.ndindex((si_y,si_x)):

    if eivec:
      iRd2, eigl,eigr= la.eig(S[:,:,j,i],left=True)
    else:
      iRd2 = la.eig(S[:,:,j,i],right=False)
  
    iRd2 = -iRd2.real
    idx = np.argsort(iRd2)
  
    iRd2 = iRd2[idx]
    with np.errstate(divide='ignore', invalid='ignore'):
      Rd_loc = 1./np.sqrt(iRd2)
    Rd[:,j,i] = Rd_loc

    if eivec:  
      eigl = eigl[:,idx]
      eigr = eigr[:,idx]
    
      # Normalize eigenvectors
      N2col = N2[:,j,i][:,None]
      cm = Rd_loc[:,None]*f0[j,i]
  
      if wmode:
        scap = np.sum(dhicol*eigr*eigr*N2col*cm.T**2,0)
      else:
        scap = np.sum(dhcol*eigr*eigr,0)
      eigr = eigr*np.sqrt(Ht/scap)*np.sign(eigr[0,:])
      
      # # scalar product
      # if wmode:
      #   check = np.sum(N2col.T*eigr[:,1]*eigr[:,1]*dhicol.T*(Rd_loc[1]*f0[j,i])**2)
      # else:
      #   check = np.sum(dhcol.T*eigr[:,2]*eigr[:,2])/Ht
  
      scap2 =  np.sum(eigl*eigr,0)
      eigl = eigl/scap2

      lay2mod[:,:,j,i] = eigl.T
      mod2lay[:,:,j,i] = eigr
  
  if eivec:  
    return Rd.squeeze(), lay2mod.squeeze(), mod2lay.squeeze()
  else:
    return Rd.squeeze()


def project_modes(psi,l2m):
  nd = psi.ndim
  if nd == 1:
    return np.einsum('ij,j->i',l2m,psi)
  elif nd == 2:
    return np.einsum('ijk,jk->ik',l2m,psi)
  elif nd == 3:
    return np.einsum('ijkl,jkl->ikl',l2m,psi)


def project_layer(psi,m2l):
  nd = psi.ndim
  if nd == 1:
    return np.einsum('ij,j->i',m2l,psi)
  elif nd == 2:
    return np.einsum('ijk,jk->ik',m2l,psi)
  elif nd == 3:
    return np.einsum('ijkl,jkl->ikl',m2l,psi)


def p2stretch(psi,dh,N2,f0):
  """
  Computes
             d  ( f^2  d     )
     q_s =   -- ( ---  -- psi)
             dz ( N^2  dz    )

  Parameters
  ----------

  psi : array [nz (,ny,nx)]
  dh : array [nz]
  N2 : array [nz (,ny,nx)]
  f0 : scalar or array [ny,nx)]

  Returns
  -------

  q_s: array [nz (,ny,nx)]
  """

  N2,f0,psi = reshape3d(dh,N2,f0,psi=psi)
  
  dhi = 0.5*(dh[1:] + dh[:-1])

  # with the inner diff, we get -b
  # we pad the outer diff with zeros (BC: d psi/dz = 0)
  q_stretch = f0**2*np.diff(np.diff(psi,1,0)/N2/dhi[:,None,None],1,0,0,0)/dh[:,None,None]
  #  qf = np.einsum('ijkl,jkl->ikl',gamma,pf)

  return q_stretch.squeeze()


def p2b(psi,dh,f0):
  """
  Computes
             d  
     b =  f  -- psi
             dz 

  Parameters
  ----------

  psi : array [nz (,ny,nx)]
  dh : array [nz]
  f0 : scalar or array [ny,nx)]

  Returns
  -------

  b: array [nz (,ny,nx)]
  """
  dhi = 0.5*(dh[1:] + dh[:-1])

  nd = psi.ndim

  if nd == 1:
    b = -f0*np.diff(psi,1,0)/dhi[:]
  elif nd == 2: 
    b = -f0*np.diff(psi,1,0)/dhi[:,None]
  elif nd == 3: 
    b = -f0*np.diff(psi,1,0)/dhi[:,None,None]

  return b.squeeze()


def laplacian(psi, Delta=1, bc='dirichlet'):

  nd = psi.ndim
  si = psi.shape

  if nd == 1:
    print("not handeling 1d arrays")
    sys.exit(1)
  elif nd == 2: 
    psi = psi.reshape(1,si[0],si[1])

  psi = pad_bc(psi, bc)

  omega = (psi[:,2:,1:-1] + psi[:,:-2,1:-1] + psi[:,1:-1,2:] + psi[:,1:-1,:-2] - 4*psi[:,1:-1,1:-1])/Delta**2

  return omega.squeeze()


def p2q(psi,dh,N2,f0,Delta,**kwargs):

  q = p2stretch(psi,dh,N2,f0) + laplacian(psi,Delta,**kwargs)

  return q

## routines from spoisson
# general poisson
def poisson2d(n, Delta=1, bc='dirichlet'):

  iDelta2 = 1/Delta**2

  n2 = n*n
  row = np.zeros(5*n2-4*n)
  col = np.zeros(5*n2-4*n)
  dat = np.zeros(5*n2-4*n)
  
  iz = -1
  for i in range(n):
    for j in range(n):
      k = i + n*j
      iz = iz + 1
      row[iz] = k
      col[iz] = k
      dat[iz] = -4*iDelta2
      iz0 = iz

      if i > 0:
        iz = iz + 1
        row[iz] = k
        col[iz] = k-1
        dat[iz] = iDelta2
      else:
        if bc == "neumann" and j == 0:
          dat[iz0] += iDelta2
      if i < n-1:
        iz = iz + 1
        row[iz] = k
        col[iz] = k+1
        dat[iz] = iDelta2
      else:
        if bc == "neumann":
          dat[iz0] += iDelta2
      if j > 0:
        iz = iz + 1
        row[iz] = k
        col[iz] = k-n
        dat[iz] = iDelta2
      else:
        if bc == "neumann":
          dat[iz0] += iDelta2
      if j < n-1:
        iz = iz + 1
        row[iz] = k
        col[iz] = k+n
        dat[iz] = iDelta2
      else:
        if bc == "neumann":
          dat[iz0] += iDelta2
              
  L = csr_matrix((dat, (row, col)), shape=(n2, n2))   
  return L


def solve(rhs, **kwargs):
  """
  Solve a linear System
  
  :param  rhs: the right hand side of the linear system (2d np array)
  :param  mat: The linear operator (optional: default: poisson)
  
  :returns: The solution of the 2d system (same shape as rhs)
  
  :raises: TODO
  """
  
  si_a = rhs.shape
  
  psi = np.array(rhs).flatten()

  n = np.int(np.sqrt(len(psi)))

  # get opt. args
  L = kwargs.get('mat', poisson2d(n, **kwargs))
    
  x = spsolve(L,psi)

  xsol = x.reshape(si_a)
  return xsol
