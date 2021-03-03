#!/usr/bin/env python

import numpy as np
import scipy.linalg as la
#from scipy.sparse import diags
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import sys

from .grid import *

# PV related stuff

def gprime2N2(dh, gprime):
  '''

  Compute the Brunt Vaisala frequency squared N2 from the reduced gravity g'


    Parameters
  ----------

  dh : array [nz]
  gprime : array [nz (,ny,nx)]

  Returns
  -------

  N2 : array[nz (,ny,nx)]
 
  '''

  if np.isscalar(gprime):
    gprime = np.array([gprime]).reshape(1,1)

  nd = gprime.ndim
  while nd < 3:
    gprime = gprime[...,None]
    nd = gprime.ndim

  dhi = 0.5*(dh[1:] + dh[:-1])
  N2 = gprime/dhi[:,None,None]

  return N2.squeeze((1,2))


def gamma_stretch(dh, N2, f0=1.0, wmode=False, squeeze=True, mat_format='dense') :
  # sqeeze is intended as an internal option
  '''
    Parameters
  ----------

  dh : array [nz]
  N2 : array [nz (,ny,nx)]
  f0 : scalar or array [(ny,nx)]
  mat_format : 'dense', 'diag', 'sym_diag'
  wmode : Bool

  Returns
  -------
  
  if mat_format == "dense"
  S: array [nz,nz (,ny,nx)]
  if mat_format == "diag"
  S: upper diagonal, main diagonal, lower diagonal, array [3, nz (,ny,nx)]
  if mat_format == "sym_diag"
  S: sqrt(lower diagonal*upper diagonal), main diagonal,[1, cumprod(sqrt(lower/upper))], array [3, nz (,ny,nx)]


  densefromdiag = spdiags(alldiags, np.array([1, 0, -1]),si_z,si_z).toarray()
  '''

  N2,f0 = reshape3d(dh,N2,f0)
  nl,si_y,si_x = N2.shape

  if wmode:
    if mat_format == 'dense':
      S = np.zeros((nl,nl,si_y,si_x))
    else:
      S = np.zeros((3,nl,si_y,si_x))
  else:
    if mat_format == 'dense':
      S = np.zeros((nl+1,nl+1,si_y,si_x))
    else:
      S = np.zeros((3,nl+1,si_y,si_x))

  sloc = np.zeros((nl+1,nl+1))
  for j,i in np.ndindex((si_y,si_x)):
  
    dhi = 0.5*(dh[1:] + dh[:-1])

    diag_p1 = np.divide(f0[j,i]**2, dhi*N2[:,j,i], out=np.zeros_like(dhi), where=N2[:,j,i]!=0)
    
    diag_m1 = diag_p1/dh[1:]
    diag_p1 = diag_p1/dh[:-1]

    # diag_p1 = f0[j,i]**2/(dh[:-1]*dhi*N2[:,j,i])
    # diag_m1 = f0[j,i]**2/(dh[1:]*dhi*N2[:,j,i])
  
    if wmode:
      diag0 = -diag_p1 - diag_m1    
      # switch diagonals on purpose for wmode
      #      diagonals = [diag0,diag_p1[1:],diag_m1[:-1]]
      if mat_format == "dense":
        np.fill_diagonal(S[:,:,j,i],diag0)
        np.fill_diagonal(S[1:,:,j,i],diag_p1[1:])
        np.fill_diagonal(S[:,1:,j,i],diag_m1[:-1])
      else:
        S[0,1:,j,i] = diag_m1[:-1] # upper diag
        S[1,:,j,i] = diag0
        S[2,:-1,j,i] = diag_p1[1:]  # lower diag
    else:
      diag0 = -np.append(diag_p1,0.)
      diag0[1:] = diag0[1:] - diag_m1
      #      diagonals = [diag0,diag_m1,diag_p1]
      if mat_format == "dense":
        np.fill_diagonal(S[:,:,j,i],diag0)
        np.fill_diagonal(S[1:,:,j,i],diag_m1)
        np.fill_diagonal(S[:,1:,j,i],diag_p1)
      else:
        S[0,1:,j,i] = diag_p1 # upper diag
        S[1,:,j,i] = diag0
        S[2,:-1,j,i] = diag_m1 # lower diag

    
  if mat_format == "sym_diag":
    # matrix is stored in upper diagonal form
    S[0,1:,:,:] = np.sqrt(S[0,1:,:,:]*S[2,:-1,:,:])
    S[0,0,:,:] = 0
    S[2,1:,:,:] = np.cumprod(S[2,:-1,:,:]/S[0,1:,:,:],0)
    S[2,0,:,:] = 1


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

def comp_modes(dh, N2, f0=1.0, eivec=False, wmode=False, diag=False):
  '''
  Compute eigenvalues (and eigenvectors) of the sturm-liouville
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


  Parameters
  ----------

  dh : array [nz]
  N2 : array [nz (,ny,nx)]
  f0 : scalar or array [(ny,nx)]
  eivec : Bool
  wmode : Bool
  diag : Bool
    Use transformation matrix to solve a symetric matrix

  Returns
  -------
  
  if eivec == T
  Rd: array [nz (,ny,nx)]
  lay2mod: array [nz,nz (,ny,nx)]
  mod2lay: array [nz,nz (,ny,nx)]

  if eivec == F
  Rd: array [nz (,ny,nx)]

  '''

  N2,f0 = reshape3d(dh,N2,f0)
  nl,si_y,si_x = N2.shape
  
  mat_format = "dense"
  if diag:
    mat_format = "sym_diag"

  S = gamma_stretch(dh,N2,f0,wmode=wmode,squeeze=False,mat_format=mat_format)

  nlt = (N2 == 0).argmax(axis=0)
  nlt = np.where(nlt == 0,nl,nlt)

  # put variables in right format
  Ht = np.cumsum(dh)
#  Ht = np.sum(dh)
  dhi = 0.5*(dh[1:] + dh[:-1])
  dhcol = dh[:,None]
  dhicol = dhi[:,None]


  if wmode:
    Rd = np.zeros((nl,si_y,si_x))
    if eivec:
      mod2lay = np.zeros((nl,nl,si_y,si_x))
      lay2mod = np.zeros((nl,nl,si_y,si_x))
  else:
    nlt = nlt + 1
    Rd = np.zeros((nl+1,si_y,si_x))
    if eivec:
      mod2lay = np.zeros((nl+1,nl+1,si_y,si_x))
      lay2mod = np.zeros((nl+1,nl+1,si_y,si_x))

  for j,i in np.ndindex((si_y,si_x)):

    if eivec:
      if diag:
        iRd2, eigs = la.eigh_tridiagonal(S[1,:nlt[j,i],j,i], S[0,1:nlt[j,i],j,i])
        eigr = S[2,:nlt[j,i],j,i,None]*eigs # D*w
        eigl = eigs/S[2,:nlt[j,i],j,i,None] # w*D^-1 if eigenvectors are stored in lines but eigl is eigl.T so we do D^-1*w
      else:
        iRd2, eigl,eigr= la.eig(S[:nlt[j,i],:nlt[j,i],j,i],left=True)
    else:
      if diag:
        iRd2 = la.eigvalsh_tridiagonal(S[1,:nlt[j,i],j,i], S[0,1:nlt[j,i],j,i])
      else:
        iRd2 = la.eig(S[:nlt[j,i],:nlt[j,i],j,i],right=False)
  
    iRd2 = -iRd2.real
    idx = np.argsort(iRd2)
  
    iRd2 = iRd2[idx]
    with np.errstate(divide='ignore', invalid='ignore'):
      Rd_loc = 1./np.sqrt(iRd2)

    Rd[:nlt[j,i],j,i] = Rd_loc

    if eivec:  
      eigl = eigl[:,idx]
      eigr = eigr[:,idx]
    
      # Normalize eigenvectors
      N2col = N2[:nlt[j,i],j,i][:,None]
      cm = Rd_loc[:nlt[j,i],None]*f0[j,i]
  
      if wmode:
        scap = np.sum(dhi[:nlt[j,i],None]*eigr*eigr*N2col*cm.T**2,0)
        Htt = Ht[nlt[j,i]]
      else:
        scap = np.sum(dh[:nlt[j,i],None]*eigr*eigr,0)
        Htt = Ht[nlt[j,i]-1]
      flip = np.sign(eigr[0,:])
      eigr = eigr*np.sqrt(Htt/scap)*flip
      

      # # scalar product
      # if wmode:
      #   check = np.sum(N2col.T*eigr[:,1]*eigr[:,1]*dhicol.T*(Rd_loc[1]*f0[j,i])**2)
      # else:
      #   check = np.sum(dhcol.T*eigr[:,2]*eigr[:,2])/Ht
  
      if diag:
        eigl = eigl/np.sqrt(Htt/scap)*flip
      else:
        scap2 =  np.sum(eigl*eigr,0)
        eigl = eigl/scap2

      lay2mod[:nlt[j,i],:nlt[j,i],j,i] = eigl.T
      mod2lay[:nlt[j,i],:nlt[j,i],j,i] = eigr
  
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
#  q_stretch = f0**2*np.diff(np.diff(psi,1,0)/N2/dhi[:,None,None],1,0,0,0)/dh[:,None,None]
  q_stretch = f0**2*np.diff(np.divide(np.diff(psi,1,0), dhi[:,None,None]*N2, out=np.zeros_like(N2), where=N2!=0),1,0,0,0)/dh[:,None,None]


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
    print("not handling 1d arrays")
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
def laplace2d(n, Delta=1, bc='dirichlet', ila2=0):
  """
  Creates a sparse matrix

             d^2    d^2
     L =     ---- + ---- + ila2
             dx^2   dy^2

  with il2 a scalar

  Computes

  Parameters
  ----------

  n : int (number of points in x)
  Delta : scalar
  bc : 'dirichlet or 'neumann'
  ila2: scalar

  Returns
  -------

  L: csr_matrix sparse matrix
  """

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
      dat[iz] = -4*iDelta2 + ila2
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
  L = kwargs.get('mat', laplace2d(n, **kwargs))
    
  x = spsolve(L,psi)

  xsol = x.reshape(si_a)
  return xsol
