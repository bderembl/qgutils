#!/usr/bin/env python

import numpy as np
import scipy.linalg as la

from .grid import *
from .pv import *

def prepare_evp(k, l, S, dqbdy, dqbdx, U, V, nu, nu4, bf):
  '''
  Compute matrices of the eigenvalue problem
  '''

  nl = len(dqbdy)
  di = np.diag_indices(nl)

  k2 = k**2 + l**2
  
  # vorticity matrix in spectral
  p2q = np.copy(S)
  p2q[di] -= k2

  diag1 = np.array(k*dqbdy - l*dqbdx,dtype=np.complex)
#  diag1 = diag1 + 1j*(k2**2*nu + k2**3*nu4) # viscosity on momentum only
  diag1[-1] = diag1[-1] + 1j*k2*bf

  diag2 = k*U + l*V
  diag2 = diag2 - 1j*(k2*nu + k2**2*nu4) # viscosity on total PV
  
  mat1 = diag2[:,None]*p2q
  mat1[di] += diag1
#  print(mat1[0,0])
#  return mat1, sparse.csc_matrix(p2q)
  return mat1, p2q


def growth_rate_kl(U, V, dh, N2, f0=1.0, beta=0, nu=0, nu4=0, bf=0, si_k = 50, kmax = 2):

  '''
  Compute growth rate of the most unstable mode in (k,l) space
  '''

  nl = len(dh)

  dqbdy = -p2stretch(U,dh,N2,f0) + beta
  dqbdx =  p2stretch(V,dh,N2,f0)

  S = gamma_stretch(dh, N2, f0)

  # get l<0 by symmetry
  si_l = int(si_k/2)
  kt = np.linspace(-kmax,kmax,si_k)
  lt = np.linspace(0.,kmax,si_l)
  
  omegai_c = np.zeros((si_l,si_k))
  eivec_c  = np.zeros((si_l,si_k,nl),dtype='complex')
  
  for il in range(0,si_l):
    for ik in range(0,si_k):
          
      mat1, mat2 = prepare_evp(kt[ik], lt[il], S, dqbdy, dqbdx, U, V, nu, nu4, bf)
      try:
        eival,eivec = la.eig(mat1,mat2)
      except:
        eivec = np.zeros((nl,nl))
        eival = np.zeros(nl)
      idx = np.argsort(np.imag(eival))[::-1]
      omegai_c[il,ik] = np.max([0.,np.imag(eival[idx])[0]])
      eivec_c[il,ik,:] = eivec[:,idx][:,0]

  return kt,lt,omegai_c,eivec_c
