#!/usr/bin/env python

import numpy as np

from .grid import *
from .pv import *
from .mgd2d import *
from .operators import *

# 

def get_w(psi,dh,N2,f0,Delta,bf=0, forcing_z=0, forcing_b=0, nu=0, nu4=0, bc_fac=0):
  '''
  Solve the omega equation with surface and bottom boundary conditions
  (cf. Vallis - chap 5.4.4)


   f^2  d  ( d    )              1  (  d                                                )  
   ---  -- ( -- w ) + del^2 w = --- (f -- (J(psi,zeta) - visc) + del^2 (-J(psi,b) + F_b)) 
   N^2  dz ( dz   )             N^2 (  dz                                               )

  if QG equations are derived from SW there are viscous terms in w 
  if QG equations are derived from PE there are no viscous terms in w


  Parameters
  ----------

  psi : array [nz,ny,nx]
  dh : array [nz]
  f0 : scalar
  N2 : array [nz]
  Delta: scalar
  bf : scalar  (bottom friction coef = d_e *f0/(2*dh[-1]) with d_e the thickness 
  of the bottom Ekman layer, or bf = Ekb/(Rom*2*dh[-1]) with non dimensional params)
  forcing_z : array [ny,nx] wind forcing (exactly the same as the rhs of the PV eq.)
  forcing_b : array [(nz,) ny,nx]  =(buoyancy forcing)/N2 (entoc)
  nu: scalar, harmonic viscosity. *if provided, only apply on the vorticity equation*
  nu4: scalar, biharmonic viscosity. *if provided, only apply on the vorticity equation*
  bc_fac: scalar (only used if psi is a node field) 0 for free slip or 1 for no slip 


  Returns
  -------

  w: array [nz-1,ny,nx]

  '''  

  f_type = field_type(psi)

  N2,f0 = reshape3d(dh,N2,f0)

  zeta = laplacian(psi, Delta, bc_fac=bc_fac)
  b = p2b(psi,dh,f0)
  psi_b = interp_on_b(psi, dh)

  jpz = jacobian(psi, zeta, Delta)
  jpb = jacobian(psi_b, b, Delta)
  
  del2z = laplacian(zeta, Delta, bc_fac=bc_fac)
  del4z = laplacian(del2z, Delta, bc_fac=bc_fac)

  rhs = p2b(jpz - nu*del2z + nu4*del4z,dh,f0) - laplacian(jpb,Delta)
  
  # boundary conditions for w
  w_bc = np.zeros_like(psi)
  w_bc[0,:,:] = forcing_z
  w_bc[-1,:,:] = -bf*zeta[-1,:,:]
  w_bc = p2b(w_bc,dh,f0)

  rhs = (rhs - w_bc)/N2

  # buoyancy forcing already divided by N2
  # TODO: if forcing_b is not 0 at the boundary, the laplacian should take it as an input 
  # and the elliptic solver should have non zero BC
  # keep it as is for now because the laplacian and the solver are consistent.
  if isinstance(forcing_b,np.ndarray):
      if forcing_b.ndim == 2:
          rhs[0] += laplacian(forcing_b, Delta)
      elif forcing_b.ndim == 3:
          rhs += laplacian(forcing_b, Delta)

  if f_type == 'node':
    rhs[...,0,:]  = 0.
    rhs[...,-1,:] = 0.
    rhs[...,:,0]  = 0.
    rhs[...,:,-1] = 0.

  w = solve_mg(rhs, Delta, "w", dh, N2, f0)

  return w
