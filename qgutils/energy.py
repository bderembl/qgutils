#!/usr/bin/env python

import numpy as np

from .grid import *
from .pv import *


def comp_vel(psi, Delta, loc='center'):

  '''
  Compute velocity at cell center or cell faces

  u = -d psi /dy
  v =  d psi /dx

  Cell center vs faces:

  +----u_f----+----u_f----+
  |           |           |
  |           |           |
 v_f  u,v_c  v_f  u,v_c  v_f
  |           |           |
  |           |           |
  +----u_f----+----u_f----+

  **warning**
  cell faces do not correspond to a C-grid:
  v_f is defined at the *eas-west* faces
  u_f is defined at the *north-south* faces

  Parameters
  ----------

  psi : array [(nz,) ny,nx]
  Delta: float
  loc: 'center' or 'faces' (default center)

  Returns
  -------
  
  size of returned arry depend on loc:
  if center -> ny,nx
  if faces -> ny+1,nx and ny,nx+1

  u: array [nz, ny(+1),nx]
  v: array [nz, ny,nx(+1)]
  '''

  psi_pad = pad_bc(psi)

  if loc == 'center':
    u = (psi_pad[...,:-2,1:-1] - psi_pad[...,2:,1:-1])/(2*Delta)
    v = (psi_pad[...,1:-1,2:] - psi_pad[...,1:-1,:-2])/(2*Delta)
  elif loc == 'faces':
    u = (psi_pad[...,:-1,1:-1] - psi_pad[...,1:,1:-1])/Delta
    v = (psi_pad[...,1:-1,1:] - psi_pad[...,1:-1,:-1])/Delta
    
  return u,v


def comp_ke(psi, Delta):

  '''
  Compute KE at cell center

  KE =  (u^2 + v^2)/2

  Parameters
  ----------

  psi : array [(nz,) ny,nx]
  Delta: float

  Returns
  -------

  KE: array [(nz,) ny,nx]

  '''

  # need to interpolate u^2 and v^2 at cell center and *not* u and v
  # So that if we compare the integral of ke and the integral of 0.5*q*p
  # we get the same answer within machine precision
  u,v = comp_vel(psi, Delta, loc='faces')

  ke = 0.25*(v[...,:,1:]**2 + v[...,:,:-1]**2 +
             u[...,1:,:]**2 + u[...,:-1,:]**2)

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
