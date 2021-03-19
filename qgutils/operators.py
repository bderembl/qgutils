#!/usr/bin/env python

import numpy as np
import sys

from .grid import *


def jacobian(p, q, Delta, pad=True):
  
  """
  Compute Arakawa Jacobian J(p,q) = dp/dx dq/dy - dp/dy dq/dx

  Parameters
  ----------

  p : array [(nz), ny,nx]
  q   : array [(nz), ny,nx]
  Delta: float
  pad: bool

  Returns
  -------

  J(p,q): array [(nz), ny,nx]
  """


  nd = p.ndim
  si = p.shape

  if nd == 1:
    print("not handeling 1d arrays")
    sys.exit(1)
  elif nd == 2: 
    p = p.reshape(1,si[0],si[1])
    q   = q.reshape(1,si[0],si[1])

  if pad:
    p = pad_bc(p, bc='dirichlet')
    q = pad_bc(q, bc='dirichlet')


  jac = ((q[:,2:,1:-1]-q[:,:-2,1:-1])*(p[:,1:-1,2:]-p[:,1:-1,:-2]) \
    +(q[:,1:-1 ,:-2]-q[:, 1:-1 ,2:])*(p[:,2:, 1:-1]-p[:,:-2, 1:-1 ]) \
    + q[:,2:, 1:-1 ]*( p[:,2:,2: ] - p[:,2:,:-2 ]) \
    - q[:,:-2, 1:-1]*( p[:,:-2,2:] - p[:,:-2,:-2]) \
    - q[:, 1:-1 ,2:]*( p[:,2:,2: ] - p[:,:-2,2: ]) \
    + q[:,1:-1 ,:-2]*( p[:,2:,:-2] - p[:,:-2,:-2]) \
    + p[:, 1:-1 ,2:]*( q[:,2:,2: ] - q[:,:-2,2: ]) \
    - p[:,1:-1 ,:-2]*( q[:,2:,:-2] - q[:,:-2,:-2]) \
    - p[:,2:, 1:-1 ]*( q[:,2:,2: ] - q[:,2:,:-2 ]) \
    + p[:,:-2, 1:-1]*( q[:,:-2,2:] - q[:,:-2,:-2]))\
    /(12.*Delta*Delta)


  return jac.squeeze()


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
