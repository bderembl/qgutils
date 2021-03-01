#!/usr/bin/env python

import numpy as np

from .grid import *
from .pv import *
from .mgd2d import *
from .operators import *

# 

def get_w(psi,dh,N2,f0,Delta,ws=None,dEk=0.):
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
  ws : array [ny,nx]
  dEk : scalar

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
     
  # missing forcing and dissipation
  rhs = p2b(jpz,dh,f0) - laplacian(jpb,Delta)
  
  if dEk != 0.:
      wb = .5*dEk * zeta[-1]
  else:
      wb = None

  rhs = rhs/N2

  w = solve_mg(rhs, Delta, "w", dh, N2, f0, ws=ws, wb=wb)

  return w
