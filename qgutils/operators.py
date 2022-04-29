#!/usr/bin/env python

import numpy as np
import sys

from .grid import *
from .pv import *


def jacobian(p, q, Delta, pad='dirichlet'):
  
  """
  Compute Arakawa Jacobian J(p,q) = dp/dx dq/dy - dp/dy dq/dx

  Parameters
  ----------

  p : array [(nz), ny,nx]
  q   : array [(nz), ny,nx]
  Delta: float
  pad: 'dirichlet', 'neumann', 'periodic', or None

  Returns
  -------

  J(p,q): array [(nz), ny ,nx] or [(nz), ny-2 ,nx-2] (if no padding)
  """

  if pad:
    p = pad_bc(p, bc=pad)
    q = pad_bc(q, bc=pad)


  jac = ((q[...,2:,1:-1]-q[...,:-2,1:-1])*(p[...,1:-1,2:]-p[...,1:-1,:-2]) \
    +(q[...,1:-1 ,:-2]-q[..., 1:-1 ,2:])*(p[...,2:, 1:-1]-p[...,:-2, 1:-1 ]) \
    + q[...,2:, 1:-1 ]*( p[...,2:,2: ] - p[...,2:,:-2 ]) \
    - q[...,:-2, 1:-1]*( p[...,:-2,2:] - p[...,:-2,:-2]) \
    - q[..., 1:-1 ,2:]*( p[...,2:,2: ] - p[...,:-2,2: ]) \
    + q[...,1:-1 ,:-2]*( p[...,2:,:-2] - p[...,:-2,:-2]) \
    + p[..., 1:-1 ,2:]*( q[...,2:,2: ] - q[...,:-2,2: ]) \
    - p[...,1:-1 ,:-2]*( q[...,2:,:-2] - q[...,:-2,:-2]) \
    - p[...,2:, 1:-1 ]*( q[...,2:,2: ] - q[...,2:,:-2 ]) \
    + p[...,:-2, 1:-1]*( q[...,:-2,2:] - q[...,:-2,:-2]))\
    /(12.*Delta*Delta)


  return jac

def jacobian_pq_ms(p, dh, N2, f0, Delta, pad='dirichlet'):
  
  """
  Compute Arakawa Jacobian J(p,q) = dp/dx dq/dy - dp/dy dq/dx
  with multiple scale formalism

  Parameters
  ----------

  p : array [(nz), ny,nx]
  dh : array [nz]
  N2 : array [nz-1 (,ny,nx)]
  f0 : scalar or array [(ny,nx)]
  Delta: float
  pad: 'dirichlet', 'neumann', 'periodic', or None

  Returns
  -------

  J(p,q): array [(nz), ny+2 ,nx+2] or [(nz), ny ,nx] (if no padding)
  """

  nl = len(dh)

  omega = laplacian(p, Delta, pad)

  jac =  jacobian(p, omega, Delta, pad)

  dhc = 0.5*(dh[1:] + dh[:-1])
  idh0 = np.zeros(nl)
  idh1 = np.zeros(nl)
  idh0[1:] = 1./(dhc*dh[1:]) 
  idh1[:-1] = 1./(dhc*dh[:-1]) 

  stretch = f0**2/N2
#  stretch = np.pad(f0**2/N2,((1,1),(0,0),(0,0)),'constant')

  j_ud = jacobian(p[:-1], p[1:] , Delta, pad)

# TODO: can be compressed if p
  jac[0] += stretch[0]*j_ud[0]*idh1[0]
  for il in range(1,nl-1):
    jac[il] += stretch[il]*j_ud[il]*idh1[il] - stretch[il-1]*j_ud[il-1]*idh0[il-1]
  jac[-1] += -stretch[-1]*j_ud[-1]*idh0[-1]


  return jac


def interp_on_b(p, dh):

  """
  Interpolate a field defined at the dynamical levels onto the buoyancy levels

  Parameters
  ----------

  p : array [nz,ny,nx]
  dh : array [nz]

  Returns
  -------

  p_b: array [nz-1, ny,nx]
  """

  # weighted average (cf. Holland 1978)
  return (p[:-1,:,:]*dh[1:,None,None] + p[1:,:,:]*dh[:-1,None,None])/(dh[:-1,None,None] + dh[1:,None,None])


def curl(u, v, Delta, mode):
  """
  Compute curl of a vector field.

  dv/dx - dudy
  
  This operation depends on the position of the vector field. 

  mode = v2c: u and v are defined on vertices and output is at cell center

  Parameters
  ----------

  u : array [(nz,)ny,nx]  (size of u and v may vary)
  v : array [(nz,)ny,nx]
  Delta: float
  mode: str, see description

  Returns
  -------

  curluv: array [(nz,) ny,nx] (horizontal dimension may vary)
  """

  if mode == "v2c":

    curluv = 0.5*(v[...,1:,1:] - v[...,1:,:-1]
                  + v[...,:-1,1:] - v[...,:-1,:-1]
                  - u[...,1:,1:]  + u[...,:-1,1:]
                  - u[...,1:,:-1] + u[...,:-1,:-1])/Delta

  return curluv
