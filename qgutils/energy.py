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


def lorenz_cycle(pfiles,dh,N2,f0,Delta,bf=0, Re=0, Re4=0, forcing=0):
  '''
  Lorenz cycle
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

  dissip_k_me = np.zeros((nl,N,N))
  dissip_p_me = np.zeros((nl,N,N))

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
  
  if Re4 !=0:
    dissip_k_me = -1/Re4*laplacian(laplacian(z_me,Delta),Delta)
    dissip_p_me = -1/Re4*laplacian(laplacian(s_me,Delta),Delta)
  if Re !=0:
    dissip_k_me += 1/Re*laplacian(z_me,Delta)
    dissip_p_me += 1/Re*laplacian(s_me,Delta)
  
  
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
  
  dissip_k = np.zeros((nl,N,N))
  dissip_p = np.zeros((nl,N,N))

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
    
    if Re4 !=0:
      dissip_k = -1/Re4*laplacian(laplacian(z_p,Delta),Delta)
      dissip_p = -1/Re4*laplacian(laplacian(s_p,Delta),Delta)
    if Re !=0:
      dissip_k += 1/Re*laplacian(z_p,Delta)
      dissip_p += 1/Re*laplacian(s_p,Delta)

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
  

  plt.figure()
  plt.text(0.5,0.5  ,"KE_m\n {0:0.0f}".format(ei_ke_me),horizontalalignment='center', verticalalignment='center',bbox=dict(boxstyle="round", fc="w"))
  plt.text(-0.5,0.5 ,"PE_m\n {0:0.0f}".format(ei_pe_me),horizontalalignment='center', verticalalignment='center',bbox=dict(boxstyle="round", fc="w"))
  plt.text(-0.5,-0.5,"PE'\n {0:0.0f}".format(np.mean(ei_pe)),horizontalalignment='center', verticalalignment='center',bbox=dict(boxstyle="round", fc="w"))
  plt.text(0.5,-0.5 ,"KE'\n {0:0.0f}".format(np.mean(ei_ke)),horizontalalignment='center', verticalalignment='center',bbox=dict(boxstyle="round", fc="w"))
  
  # wb
  wb_sign = np.sign(ei_wb_me)
  plt.arrow(-wb_sign*0.25,0.5,wb_sign*0.5,0,width=0.01)
  plt.text(0,0.55, "{0:0.0f}".format(np.abs(ei_wb_me)),horizontalalignment='center', verticalalignment='center')
  
  wb_sign = np.sign(np.mean(ei_wb))
  plt.arrow(-wb_sign*0.25,-0.5,wb_sign*0.5,0,width=0.01)
  plt.text(0,-0.45, "{0:0.0f}".format(np.abs(np.mean(ei_wb))),horizontalalignment='center', verticalalignment='center')
  
  # mean to eddy
  k2k_sign = np.sign(np.mean(ei_ke_me2ke_p))
  plt.arrow(0.5,k2k_sign*0.3,0,-k2k_sign*0.5,width=0.01)
  plt.text(0.6,0, "{0:0.0f}".format(np.abs(np.mean(ei_ke_me2ke_p))))
  
  p2p_sign = np.sign(np.mean(ei_pe_me2pe_p))
  plt.arrow(-0.5,p2p_sign*0.3,0,-p2p_sign*0.5,width=0.01)
  plt.text(-0.8,0, "{0:0.0f}".format(np.abs(np.mean(ei_pe_me2pe_p))))
  
  
  # forcing
  plt.arrow(0.5,1.25,0,-0.5,width=0.01)
  plt.text(0.55,1, "ws:{0:0.0f}".format(ei_surf_me))
  
  # viscous dissip
  plt.arrow(0.75,0.6,0.5,0,width=0.01)
  plt.text(0.75,0.65, "D:{0:0.0f}".format(-ei_diss_k_me))
  
  plt.arrow(0.75,-0.4,0.5,0,width=0.01)
  plt.text(0.75,-0.35, "D:{0:0.0f}".format(-np.mean(ei_diss_k)))
  
  plt.arrow(-0.75,0.5,-0.5,0,width=0.01)
  plt.text(-1.25,0.55, "D:{0:0.0f}".format(-ei_diss_p_me))
  
  plt.arrow(-0.75,-0.5,-0.5,0,width=0.01)
  plt.text(-1.25,-0.45, "D:{0:0.0f}".format(-np.mean(ei_diss_p)))
  
  # Bottom friction
  plt.arrow(0.75,0.4,0.5,0,width=0.01)
  plt.text(0.75,0.3, "BF:{0:0.0f}".format(-ei_bottom_me))
  
  plt.arrow(0.75,-0.6,0.5,0,width=0.01)
  plt.text(0.75,-0.7, "BF:{0:0.0f}".format(-np.mean(ei_bottom)))
  
  
  plt.xlim([-1.5,1.5])
  plt.ylim([-1.5,1.5])
       
  plt.show()
