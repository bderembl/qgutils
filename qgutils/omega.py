#!/usr/bin/env python

import numpy as np

from .pv import *

# 

def get_w(psi,dh,N2,f0,Delta):
  '''
  Solve the omega equation without forcing and dissipation 
  (cf. Vallis - chap 5.4.4)


     f^2   d  (  d     )                   d     
     ---   -- (  -- psi)  + del^2 psi =  f -- J(psi,zeta)  -del^2 J(psi,b) 
     N^2   dz (  dz    )                   dz    


  Parameters
  ----------

  psi : array [nz,ny,nx]
  dh : array [nz]
  f0 : scalar
  N2 : array [nz]
  Delta: scalar

  Returns
  -------

  w: array [nz -1, ny,nx]

  '''  


  N2,f0 = reshape3d(dh,N2,f0)

  zeta = laplacian(psi, Delta)
  b = p2b(psi,dh,f0)
  psi_b = interp_on_b(psi, dh)

  jpz = jacobian(psi, zeta, Delta)
  jpb = jacobian(psi_b, b, Delta)
     
  # missing forcing and dissipation
  rhs = p2b(jpz,dh,f0) - laplacian(jpb,Delta)

  rhs = rhs/N2

  w = solve_mg(rhs, Delta, "w", dh, N2, f0)

  return w
