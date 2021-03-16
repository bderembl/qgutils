#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from .grid import *
from .pv import *
from .omega import *
from .inout import *


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
  N2 : array [nz, (ny,nx)]
  f0 : scalar or array [ny,nx]
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


def integral_plev(psi, dh, Delta, average=False):

  '''
  Compute integral of a field defined at a p-level


  Parameters
  ----------

  psi : array [nz, ny,nx]
  dh : array [nz]
  Delta: float
  average: if True: divide the integral by the total volume (default is False)

  Returns
  -------

  psi_i = scalar

  '''
  si = psi.shape
  N = si[-1]

  Ht = np.sum(dh)

  if average:
    psi_i = np.sum(psi*dh[:,None,None])/Ht/N**2
  else:
    psi_i = np.sum(psi*dh[:,None,None])*Delta*Delta

  return psi_i


def integral_blev(psi, dh, Delta, average=False):

  '''
  Compute integral of a field defined at a b-level


  Parameters
  ----------

  psi : array [nz-1, ny,nx]
  dh : array [nz] **dh is the layer thickness of the p-level**
  Delta: float
  average: if True: divide the integral by the total volume (default is False)

  Returns
  -------

  psi_i = scalar

  '''

  si = psi.shape
  N = si[-1]

  dhi = 0.5*(dh[1:] + dh[:-1])
  Ht = np.sum(dh) # we still divide by the total thickness (assume b=0 at top and
                  # bottom)

  if average:
    psi_i = np.sum(psi*dhi[:,None,None])/Ht/N**2
  else:
    psi_i = np.sum(psi*dhi[:,None,None])*Delta*Delta

  return psi_i


def lorenz_cycle(pfiles,dh,N2,f0,Delta,bf=0, nu=0, nu4=0, forcing=0):
  '''
  Compute Lorenz energy cycle

  Parameters
  ----------

  pfiles : list of pressure files
  dh : array [nz] 
  N2 : array [nz (,ny,nx)]
  f0 : scalar or array [ny,nx]
  Delta: float
  bf : scalar  (bottom friction coef = d_e *f0/(2*dh[-1]) with d_e the thickness 
  of the bottom Ekman layer, or bf = Ekb/(Rom*2*dh[-1]) with non dimensional params)
  nu: scalar, harmonic viscosity
  nu4: scalar, bi-harmonic viscosity
  forcing : array [ny,nx] (exactly the same as the rhs of the PV eq.)

  Returns
  -------

  lec: dict of all energy fluxes and energy reservoirs. Sign convention matches name:
    e.g. if mke2mpe >0 then there is a transfer from mke to mpe
  '''

  nf  = len(pfiles)
  N2,f0 = reshape3d(dh,N2,f0)

  p = np.fromfile(pfiles[0],'f4')
  N = int(p[0])
  N1 = N + 1
  nl = int(len(p)/N1**2)
  
  # compute mean
  p_me = np.zeros((nl,N,N))
  w_me = np.zeros((nl-1,N,N))

  for ifi in range(0,nf):
  
    p = read_bas(pfiles[ifi])
    w = get_w(p,dh, N2[:,0,0],f0[0,0], Delta, bf,forcing)
  
    p_me += p
    w_me += w
  
  
  p_me /= nf
  w_me /= nf
  
  z_me = laplacian(p_me,Delta)
  b_me = p2b(p_me, dh, f0)
  s_me = p2stretch(p_me,dh, N2,f0)
  q_me = p2q(p_me, dh, N2,f0, Delta)
  ke_me = comp_ke(p_me,Delta)
  pe_me = comp_pe(p_me, dh, N2,f0, Delta)
  
  ei_ke_me = integral_plev(ke_me, dh, Delta)
  ei_pe_me = integral_blev(pe_me, dh, Delta)
  
  e_surf   = np.zeros((nl,N,N))
  e_bottom = np.zeros((nl,N,N))
  
  dissip_k_me = -nu4*laplacian(laplacian(z_me,Delta),Delta)
  dissip_p_me = -nu4*laplacian(laplacian(s_me,Delta),Delta)
  dissip_k_me += nu*laplacian(z_me,Delta)
  dissip_p_me += nu*laplacian(s_me,Delta)
  
  
  bottom_ekman = -bf*laplacian(p_me[-1,:,:],Delta)
  
  e_surf[0,:,:] = -p_me[0,:,:]*forcing
  e_bottom[-1,:,:] = -p_me[-1,:,:]*bottom_ekman
  
  ei_surf_me   = integral_plev(e_surf, dh, Delta)
  ei_bottom_me = integral_plev(e_bottom, dh, Delta)
  ei_diss_k_me = integral_plev(-p_me*dissip_k_me, dh, Delta)
  ei_diss_p_me = integral_plev(-p_me*dissip_p_me, dh, Delta)
  ei_wb_me     = integral_blev(w_me*b_me, dh, Delta)
  
  # compute all terms
  ei_ke   = np.zeros(nf)
  ei_pe   = np.zeros(nf)
  ei_surf   = np.zeros(nf)
  ei_bottom = np.zeros(nf)
  ei_diss_k = np.zeros(nf)
  ei_diss_p = np.zeros(nf)
  ei_wb     = np.zeros(nf)
  ei_ke_me2ke_p = np.zeros(nf)
  ei_pe_me2pe_p = np.zeros(nf)
  
  for ifi in range(0,nf):
    
    p = read_bas(pfiles[ifi])
  
    z = laplacian(p,Delta)
    b = p2b(p, dh, f0)
    s = p2stretch(p,dh, N2,f0)
    w = get_w(p,dh, N2[:,0,0],f0[0,0], Delta, bf,forcing)
    q = p2q(p, dh, N2,f0, Delta)
    ke = comp_ke(p,Delta)
    pe = comp_pe(p, dh, N2,f0, Delta)
  
    p_p = p - p_me
    z_p = z - z_me
    b_p = b - b_me 
    s_p = s - s_me
    w_p = w - w_me
    q_p = q - q_me
    ke_p = ke - ke_me
    pe_p = pe - pe_me
  
    jpz = jacobian(p_p,z_p, Delta)
    jps = jacobian(p_p,s_p, Delta)
  
    ke_me2ke_p = -p_me*jpz
    pe_me2pe_p = -p_me*jps
    
    dissip_k = -nu4*laplacian(laplacian(z_p,Delta),Delta)
    dissip_p = -nu4*laplacian(laplacian(s_p,Delta),Delta)
    dissip_k += nu*laplacian(z_p,Delta)
    dissip_p += nu*laplacian(s_p,Delta)

    bottom_ekman = -bf*laplacian(p_p[-1,:,:],Delta)
  
    e_surf[0,:,:] = -p_p[0,:,:]*forcing
    e_bottom[-1,:,:] = -p_p[-1,:,:]*bottom_ekman
  
    ei_ke_me2ke_p[ifi] = integral_plev(ke_me2ke_p, dh, Delta) 
    ei_pe_me2pe_p[ifi] = integral_plev(pe_me2pe_p, dh, Delta) 
    ei_surf[ifi]   = integral_plev(e_surf, dh, Delta)
    ei_bottom[ifi] = integral_plev(e_bottom, dh, Delta)
    ei_diss_k[ifi] = integral_plev(-p_p*dissip_k, dh, Delta)
    ei_diss_p[ifi] = integral_plev(-p_p*dissip_p, dh, Delta)
    ei_wb[ifi]     = integral_blev(w_p*b_p, dh, Delta)
    ei_ke[ifi]     = integral_plev(ke_p, dh, Delta)
    ei_pe[ifi]     = integral_blev(pe_p, dh, Delta)
  

  # sign convention matches name
  lec = {}
  lec["f2mke"]   = ei_surf_me             
  lec["mke2mpe"] = -ei_wb_me              
  lec["epe2eke"] = np.mean(ei_wb)         
  lec["mke2eke"] = np.mean(ei_ke_me2ke_p) 
  lec["mpe2epe"] = np.mean(ei_pe_me2pe_p) 
  lec["mke2dis"] = -ei_diss_k_me          
  lec["eke2dis"] = -np.mean(ei_diss_k)    
  lec["mpe2dis"] = -ei_diss_p_me          
  lec["epe2dis"] = -np.mean(ei_diss_p)    
  lec["mke2bf"]  = -ei_bottom_me          
  lec["eke2bf"]  = -np.mean(ei_bottom)    
  lec["mke"]     = ei_ke_me      
  lec["eke"]     = np.mean(ei_ke)
  lec["mpe"]     = ei_pe_me      
  lec["epe"]     = np.mean(ei_pe)

  return lec


def draw_lorenz_cycle(lec):

  '''
  Draw Lorenz energy cycle

  Parameters
  ----------

  lec: dict of all energy fluxes and energy reservoirs from lorenz_cycle function

  Returns
  -------

  nothing

  '''

  f2mke   = lec["f2mke"]  
  mke2mpe = lec["mke2mpe"]
  epe2eke = lec["epe2eke"]
  mke2eke = lec["mke2eke"]
  mpe2epe = lec["mpe2epe"]
  mke2dis = lec["mke2dis"]
  eke2dis = lec["eke2dis"]
  mpe2dis = lec["mpe2dis"]
  epe2dis = lec["epe2dis"]
  mke2bf  = lec["mke2bf"] 
  eke2bf  = lec["eke2bf"] 
  mke     = lec["mke"]    
  eke     = lec["eke"]    
  mpe     = lec["mpe"]    
  epe     = lec["epe"]

  plt.figure()
  plt.text(0.5,0.5  ,"MKE\n {0:0.0f}".format(mke),horizontalalignment='center', verticalalignment='center',bbox=dict(boxstyle="round", fc="w"))
  plt.text(-0.5,0.5 ,"MPE\n {0:0.0f}".format(mpe),horizontalalignment='center', verticalalignment='center',bbox=dict(boxstyle="round", fc="w"))
  plt.text(-0.5,-0.5,"EPE\n {0:0.0f}".format(epe),horizontalalignment='center', verticalalignment='center',bbox=dict(boxstyle="round", fc="w"))
  plt.text(0.5,-0.5 ,"EKE\n {0:0.0f}".format(eke),horizontalalignment='center', verticalalignment='center',bbox=dict(boxstyle="round", fc="w"))
  
  # wb
  wb_sign = np.sign(mke2mpe)
  plt.arrow(wb_sign*0.25,0.5,-wb_sign*0.5,0,width=0.01, length_includes_head=True)
  plt.text(0,0.6, "{0:0.0f}".format(np.abs(mke2mpe)),horizontalalignment='center', verticalalignment='center')
  
  wb_sign = np.sign(epe2eke)
  plt.arrow(-wb_sign*0.25,-0.5,wb_sign*0.5,0,width=0.01, length_includes_head=True)
  plt.text(0,-0.4, "{0:0.0f}".format(np.abs(epe2eke)),horizontalalignment='center', verticalalignment='center')
  
  # mean to eddy
  k2k_sign = np.sign(mke2eke)
  plt.arrow(0.5,k2k_sign*0.25,0,-k2k_sign*0.5,width=0.01, length_includes_head=True)
  plt.text(0.6,0, "{0:0.0f}".format(np.abs(mke2eke)))
  
  p2p_sign = np.sign(mpe2epe)
  plt.arrow(-0.5,p2p_sign*0.25,0,-p2p_sign*0.5,width=0.01, length_includes_head=True)
  plt.text(-0.8,0, "{0:0.0f}".format(np.abs(mpe2epe)))
  
  
  # forcing
  plt.arrow(0.5,1.25,0,-0.5,width=0.01, length_includes_head=True)
  plt.text(0.55,1, "ws:{0:0.0f}".format(f2mke))
  
  # viscous dissip
  plt.arrow(0.75,0.6,0.5,0,width=0.01, length_includes_head=True)
  plt.text(1,0.65, "D:{0:0.0f}".format(mke2dis),horizontalalignment='center')
  
  plt.arrow(0.75,-0.4,0.5,0,width=0.01)
  plt.text(1,-0.35, "D:{0:0.0f}".format(eke2dis),horizontalalignment='center')
  
  plt.arrow(-0.75,0.5,-0.5,0,width=0.01, length_includes_head=True)
  plt.text(-1,0.55, "D:{0:0.0f}".format(mpe2dis),horizontalalignment='center')
  
  plt.arrow(-0.75,-0.5,-0.5,0,width=0.01, length_includes_head=True)
  plt.text(-1,-0.45, "D:{0:0.0f}".format(epe2dis),horizontalalignment='center')
  
  # Bottom friction
  plt.arrow(0.75,0.4,0.5,0,width=0.01, length_includes_head=True)
  plt.text(1,0.25, "BF:{0:0.0f}".format(mke2bf),horizontalalignment='center')
  
  plt.arrow(0.75,-0.6,0.5,0,width=0.01, length_includes_head=True)
  plt.text(1,-0.75, "BF:{0:0.0f}".format(eke2bf),horizontalalignment='center')
  
  
  plt.xlim([-1.5,1.5])
  plt.ylim([-1.5,1.5])
       
  plt.show()

  return
