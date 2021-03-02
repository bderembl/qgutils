#!/usr/bin/env python

import numpy as np

from .grid import *
from .pv import *
from .mgd2d import *
from .operators import *

# 

def get_w(psi,dh,N2,f0,Delta,bf=0, forcing=0):
  '''
  Solve the omega equation with surface and bottom boundary conditions
  (cf. Vallis - chap 5.4.4)


     f^2   d  (  d     )                 1  (  d                              )  
     ---   -- (  -- psi)  + del^2 psi = --- (f -- J(psi,zeta)  -del^2 J(psi,b)) 
     N^2   dz (  dz    )                N^2 (  dz                             )


  Parameters
  ----------

  psi : array [nz,ny,nx]
  dh : array [nz]
  f0 : scalar
  N2 : array [nz]
  Delta: scalar
  bf : scalar  (bottom friction coef = Ekb/(Rom*2*dh[-1]) )
  forcing : array [ny,nx] (exactly the same as the rhs of the PV eq.)

  Returns
  -------

  w: array [nz-1,ny,nx]

  '''  


  N2,f0 = reshape3d(dh,N2,f0)

  zeta = laplacian(psi, Delta)
  b = p2b(psi,dh,f0)
  psi_b = interp_on_b(psi, dh)

  jpz = jacobian(psi, zeta, Delta)
  jpb = jacobian(psi_b, b, Delta)
     
  rhs = p2b(jpz,dh,f0) - laplacian(jpb,Delta)
  
  # boundary conditions for w
  w_bc = np.zeros_like(psi)
  w_bc[0,:,:] = forcing
  w_bc[-1,:,:] = -bf*zeta[-1,:,:]
  w_bc = p2b(w_bc,dh,f0)

  rhs = (rhs - w_bc)/N2

  w = solve_mg(rhs, Delta, "w", dh, N2, f0)

  return w
