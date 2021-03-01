#!/usr/bin/env python

import numpy as np

from .grid import *
from .pv import *


def comp_vel(psi, Delta):

  '''
  Compute velocity at cell center

  u = -d psi /dy
  v =  d psi /dx

  '''

  psi_pad = pad_bc(psi)

  u = (psi_pad[:,:-2,1:-1] - psi_pad[:,2:,1:-1])/(2*Delta)
  v = (psi_pad[:,1:-1,2:] - psi_pad[:,1:-1,:-2])/(2*Delta)

  return u,v


def comp_ke(psi, Delta):

  '''
  Compute KE at cell center

  KE =  (u^2 + v^2)/2

  Parameters
  ----------

  psi : array [nz, ny,nx]
  Delta: float

  Returns
  -------

  KE: array [nz, ny,nx]

  '''

  nd = psi.ndim
  si = psi.shape
  N = si[-1]
  if nd != 3:
    print("dimension of psi should be [nz, ny,nx]")
    sys.exit(1)

  psipad = pad_bc(psi)
  v0 = np.diff(psipad,1,axis=2)/Delta
  #v = 0.5*(v0[:,1:-1,1:] + v0[:,1:-1,:-1])

  u0 = -np.diff(psipad,1,axis=1)/Delta
  #u = 0.5*(u0[:,1:,1:-1] + u0[:,:-1,1:-1])

  # need to interpolate u^2 and v^2 and *not* u and v
  # So that if we compare the integral of ke and the integral of 0.5*q*p
  # we get the same answer within machine precision
  ke_v = 0.5*0.5*(v0[:,1:-1,1:]**2 + v0[:,1:-1,:-1]**2)
  ke_u = 0.5*0.5*(u0[:,1:,1:-1]**2 + u0[:,:-1,1:-1]**2)

  ke = ke_u + ke_v

  return ke


def comp_pe(psi, dh,N2,f0,Delta):

  '''
  Compute KE at cell center

  PE =  b^2/(2 N^2)

  Parameters
  ----------

  psi : array [nz, ny,nx]
  dh : array [nz]
  f0 : scalar or array [ny,nx]
  N2 : array [nz, (ny,nx)]
  Delta: float

  Returns
  -------

  PE: array [nz-1, ny,nx]

  '''

  nd = psi.ndim
  si = psi.shape
  N = si[-1]
  if nd != 3:
    print("dimension of psi should be [nz, ny,nx]")
    sys.exit(1)

  N2,f0 = reshape3d(dh,N2,f0)

  b = p2b(psi,dh,f0)
  pe = 0.5*b**2/N2

  return pe
