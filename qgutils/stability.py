#!/usr/bin/env python

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

import os.path

from .grid import *
from .pv import *

def prepare_evp(k, l, S, dqbdy, dqbdx, U, V, nu=0, nu4=0, bf=0, sd=0, mat_format='dense'):
  '''
  Compute matrices of the eigenvalue problem
  '''

  nl = len(dqbdy)
  if mat_format == 'dense':
    di = np.diag_indices(nl)
  else:
    di = (np.ones((nl,), dtype=int), np.arange(nl))

  k2 = k**2 + l**2
  
  # vorticity matrix in spectral
  p2q = np.copy(S)
  p2q[di] -= k2

  diag1 = np.array(k*dqbdy - l*dqbdx,dtype=np.complex)
#  diag1 = diag1 + 1j*(k2**2*nu + k2**3*nu4) # viscosity on momentum only
  diag1[-1] += 1j*k2*bf

  diag2 = k*U + l*V
  diag2 = diag2 - 1j*(k2*nu + k2**2*nu4) # viscosity on total PV

#  diag2[0] -= 1j*sd # thermal damping on total pv
  
  if mat_format == 'dense':
    mat1 = diag2[:,None]*p2q
    mat1[di] += diag1
    #  mat1[0,0:2] -= 1j*sd*S[0,0:2]
  else:
    mat1 = np.zeros((3,nl),dtype=np.complex)
    mat1[0,1:]  = diag2[:-1]*p2q[0,1:]
    mat1[1,:]   = diag2*p2q[1,:] + diag1
    mat1[2,:-1] = diag2[1:]*p2q[2,:-1]

  return mat1, p2q


def growth_rate_kl(U, V, dh, N2, f0=1.0, beta=0, nu=0, nu4=0, bf=0, sd=0, si_k = 50, kmax = 1):

  '''
  Compute growth rate of the most unstable mode in (k,l) space

  Parameters
  ----------

  U : large-scale zonal velocity [nz]
  U : large-scale meridional velocity [nz]
  dh: layer thickness [nz]
  N2 : brunt vaisala frequency [nz-1]
  f0 : coriolis parameter
  beta: beta 
  nu: harmonic viscosity
  nu4: bi-harmonic viscosity
  bf: bottom friction
  sd: surface damping
  si_k: number of grid points in spectral space
  kmax: size of the spectral window

  Returns
  -------
  
  kt: zonal wave number (cycle per unit length) [si_k]
  lt: meridional wave number [si_k/2]
  omega: growth rate of the most unstable mode [sik/2,si_k]
  eigenvectors: of the most unstable mode [sik/2,si_k,nz]

  '''

  diag = False
  if diag:
    print("*** diag=True does not work well, do not use ***")
    mat_format='diag'
  else:
    mat_format='dense'

  nl = len(dh)

  nlt = (N2 == 0).argmax(axis=0)
  if nlt == 0:
    nlt = nl
  nlt = nlt + 1

  dqbdy = -p2stretch(U,dh,N2,f0) + beta
  dqbdx =  p2stretch(V,dh,N2,f0)

  S = gamma_stretch(dh, N2, f0, mat_format=mat_format)

  # get l<0 by symmetry
  si_l = int(si_k/2)
  kt = np.linspace(-kmax,kmax,si_k)
  lt = np.linspace(0.,kmax,si_l)
  
  omegai_c = np.zeros((si_l,si_k))
  eivec_c  = np.zeros((si_l,si_k,nl),dtype='complex')

  eivec = np.zeros(nl,dtype=complex) + np.random.random(nl)
  for il in range(0,si_l):
    for ik in range(0,si_k):
          
      mat1, mat2 = prepare_evp(2*np.pi*kt[ik], 2*np.pi*lt[il], S, dqbdy, dqbdx, U, V, nu, nu4, bf, sd, mat_format=mat_format)

      if diag:
        mat1 = sp.spdiags(mat1, np.array([1, 0, -1]),nlt,nlt)
        mat2 = sp.spdiags(mat2, np.array([1, 0, -1]),nlt,nlt)

        # def invq_spec(rhs):
        #   x = la.solve_banded((1,1),mat2,rhs)
        #   return x

        # M = sp.linalg.LinearOperator((nl, nl),  matvec = invq_spec )
        try:
          eival,eivec = sp.linalg.eigs(sp.linalg.inv(mat2)*mat1,k=1,v0=eivec, which='LI')
        except:
          eivec = np.zeros(nlt,dtype=complex) + np.random.random(nlt)
          eival = 1j
        omegai_c[il,ik] = np.max([0.,np.imag(eival)])
        eivec_c[il,ik,:nlt] = np.squeeze(eivec)
      else:
        try:
          eival,eivec = la.eig(mat1[:nlt,:nlt],mat2[:nlt,:nlt])
        except:
          eivec = np.zeros((nlt,nlt))
          eival = np.zeros(nlt)
        idx = np.argmax(np.imag(eival))
        omegai_c[il,ik] = np.max([0.,np.imag(eival[idx])])
        eivec_c[il,ik,:nlt] = eivec[:nlt,idx]

  return kt,lt,omegai_c,eivec_c


def init_file_stability(filename, U, V, dh, N2, f0=1.0, beta=0, nu=0, nu4=0, bf=0, sd=0, si_k = 50, kmax = 1, omin = 1e-7):

  '''
  Create a pickle file to backup the stability analysis

  Parameters
  ----------

  filename : path of backup. If file already exists, don't do anything
  U : large-scale zonal velocity [nz, (ny,nx)]
  U : large-scale meridional velocity [nz, (ny,nx)]
  dh: layer thickness [nz]
  N2 : brunt vaisala frequency [nz-1, (ny,nx)]
  f0 : coriolis parameter (scalar or [(ny,nx)])
  beta: beta 
  nu: harmonic viscosity
  nu4: bi-harmonic viscosity
  bf: bottom friction
  sd: surface damping
  si_k: number of grid points in spectral space
  kmax: size of the spectral window (scalar or [(ny,nx)])
  omin: minimum threshold frequency for detecting instability

  '''
  if os.path.exists(filename):
    print('file already exists: skipping')
  else:
    nd = U.ndim
    while nd < 3:
      U = U[...,None]
      V = V[...,None]
      N2 = N2[...,None]
      if np.isscalar(f0):
        f0 = np.array([f0]).reshape(1,1)
      else:
        f0 = f0[...,None]
  
    nl,si_y,si_x = U.shape
  
    kmax_arr = kmax + np.zeros((si_y,si_x))
    conv = np.zeros((si_y,si_x))
    ku = np.zeros((si_y,si_x)) + np.nan
    lu = np.zeros((si_y,si_x)) + np.nan
    ou = np.zeros((si_y,si_x)) + np.nan
    omegai_c = np.zeros((si_k,si_k,si_y,si_x))
  
    instab_data = {'U':U, 'V': V, 'dh':dh, 'N2':N2, 'f0':f0, 'beta':beta, 'nu':nu, 'nu4':nu4, 'bf':bf, 'sd':sd, 'si_k':si_k, 'omin':omin, 'ku':ku, 'lu':lu, 'ou':ou, 'kmax':kmax_arr, 'omegai_c':omegai_c, 'conv':conv}
  
  
    np.save(filename, instab_data)


def loop_growthrate(filename, flag_print=False):

  '''
  Loop over grid points in pickle file and do linear stability analysis

  Parameters
  ----------

  filename : path of backup
  flag_print : Bool

  '''

  instab_data = np.load(filename,allow_pickle=True).item()
  
  nl,si_y,si_x = instab_data['U'].shape

  nco = 0
  for j,i in np.ndindex((si_y,si_x)):

    if instab_data['conv'][j,i] < 1:

      nco += 1
      # periodic save
      if nco%60 == 0:
        np.save(filename, instab_data)

      kt,lt,omegai_c,eivec_c = growth_rate_kl(instab_data['U'][:,j,i], instab_data['V'][:,j,i], instab_data['dh'][:], instab_data['N2'][:,j,i], instab_data['f0'][j,i], instab_data['beta'], instab_data['nu'], instab_data['nu4'], instab_data['bf'], instab_data['sd'], instab_data['si_k'], instab_data['kmax'][j,i])
  
      (ilmax,ikmax) = np.unravel_index(omegai_c[:,:].argmax(), omegai_c[:,:].shape)
        
      fw = np.int(instab_data['si_k']/4)
    
      om_bdy0 = np.sum(omegai_c[-1,:]) + np.sum(omegai_c[:,0]) + np.sum(omegai_c[:,-1])
      om_bdy1 = np.sum(omegai_c[-1-fw,:]) + np.sum(omegai_c[:,+fw]) + np.sum(omegai_c[:,+fw])
      
      instab_data['conv'][j,i] = 0
    
      if ikmax > 0 and ikmax < instab_data['si_k']-1:
        instab_data['ku'][j,i] = kt[ikmax]
        instab_data['lu'][j,i] = lt[ilmax]
        instab_data['ou'][j,i] = omegai_c[ilmax,ikmax]
        instab_data['conv'][j,i] = 1
        
        if np.abs(om_bdy0) > instab_data['omin']:
          instab_data['conv'][j,i] = -1 # enlarge window
          instab_data['kmax'][j,i] = 1.2*instab_data['kmax'][j,i]
        elif np.abs(om_bdy1) <= instab_data['omin']:
          instab_data['conv'][j,i] = -2 # reduce window
          instab_data['kmax'][j,i] = 0.8*instab_data['kmax'][j,i]
      else:
        if np.sum(omegai_c[:,:]) == 0: # no instability
          instab_data['conv'][j,i] = 2
        else:
          instab_data['conv'][j,i] = -3 # enlarge window
          instab_data['kmax'][j,i] = 1.5*instab_data['kmax'][j,i]
          
      if flag_print:
        print ('({0},{1}): conv {2}'.format(j,i, instab_data['conv'][j,i]))
  
  # final save
  np.save(filename, instab_data)
