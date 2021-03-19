#!/usr/bin/env python

import numpy as np
import scipy.io.netcdf as netcdf

from .grid import *

def write_bas(fname, psi):
  """
  Write basilisk format output

  Parameters
  ----------

  psi : array [(nz,) ny,nx]
  fname: file name

  Returns
  -------

  nothing
  """

  nd = psi.ndim

  if nd == 1:
    print("not handeling 1d arrays")
    sys.exit(1)
  elif nd == 2:
    psi = psi[None,:,:]

  nl,N,naux = psi.shape

  # combine and output in .bas format: (z,x,y)
  p_out = np.zeros((nl,N+1,N+1))
  p_out[:,0,0] = N
  p_out[:,1:,1:] = psi
  p_out = np.transpose(p_out,(0,2,1))
  p_out.astype('f4').tofile(fname)


def read_bas(fname):
  """
  Read basilisk format file

  Parameters
  ----------

  fname: file name

  Returns
  -------

  psi : array [(nz,) ny,nx]
  """

  psi  = np.fromfile(fname,'f4')
  N = int(psi[0])
  N1 = N + 1
  nl = int(len(psi)/N1**2)

  psi  = psi.reshape(nl,N+1,N+1).transpose(0,2,1)[:,1:,1:]

  return psi.squeeze()


def read_qgcm(fname, it, var='p', rescale=1.0, interp=True, subtract_bc=False):
  """
  Read q-gcm format file

  Parameters
  ----------

  fname: file name
  it: int iteration number
  var: str, variable name (optional)
  rescale: float, multiply output by rescale factor
  interp: Bool, interpolate on grid center (default: True)

  Returns
  -------

  psi : array [nz, ny,nx]
  """

  f = netcdf.netcdf_file(fname,'r')
  psi = f.variables[var][it,...].copy()
  f.close()

  if subtract_bc:
    psi = psi - psi[:,0,0][:,None,None]

  psi *= rescale

  if interp:
    psi = interp_on_c(psi)

  return psi


def read_time(pfiles):
  """
  Check number of time stemps

  Parameters
  ----------

  pfiles : list of pressure files

  Returns
  -------

  si_t : int, total number of timestep
  """

  if pfiles[0][-4:] == '.bas':
    si_t = len(pfiles)
  else:
    f = netcdf.netcdf_file(pfiles[0],'r')
    time = f.variables['time'][:].copy()
    f.close()
    si_t = int(len(time)*len(pfiles))
  return si_t


def load_generic(pfiles, it, var='p', rescale=1, interp=False, si_t=1, subtract_bc=False):
  """
  Generic load function for list of files

  Parameters
  ----------

  pfiles : list of pressure files
  it: int iteration number
  var: str, variable name (optional)
  rescale: float, multiply output by rescale factor
  interp: Bool, interpolate on grid center (default: False)
  si_t: int, total length of data set
  subtract_bc: Bool, subtract bc (default: False)

  Returns
  -------

  psi : array [nz, ny,nx]
  """

  if pfiles[0][-4:] == '.bas':
    p = read_bas(pfiles[it])
  else:
    it_per_file = int(si_t/len(pfiles))
    ifi = int(it/it_per_file)
    it1 = int(it - ifi*it_per_file)
    p = read_qgcm(pfiles[ifi], it1, var, rescale, interp, subtract_bc)

  return p
