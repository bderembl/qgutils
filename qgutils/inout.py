#!/usr/bin/env python

import numpy as np
from scipy.io import netcdf_file

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


def read_nc(fname, it, var='p', rescale=1.0, interp=False, subtract_bc=False):
  """
  Read netcdf format file

  Parameters
  ----------

  fname: file name
  it: int iteration number
  var: str, variable name (optional)
  rescale: float, multiply output by rescale factor
  interp: Bool, interpolate on grid center (default: False) (changed default 7/12/22)

  Returns
  -------

  psi : array [nz, ny,nx]
  """

  f = netcdf_file(fname,'r')
  psi = f.variables[var][it,...].copy()
  f.close()

  if subtract_bc:
    psi = psi - psi[:,0,0][:,None,None]

  psi *= rescale

  if interp:
    psi = interp_on_c(psi)

  return psi


def write_nc(fname, var, timeDim = False):
  """
  write netcdf format file

  Parameters
  ----------

  fname: file name
  it: int iteration number
  var: dictionnary of variables. variables are typically 2d or 3d variables
  timeDim: bool  add an extra time dimension (default = False)

  Returns
  -------

  Nothing
  """

  f = netcdf_file(fname,'w')

  # assume is the same for all fields
  for key, psi in var.items():
    nd = psi.ndim
    si = psi.shape

    nd0 = 0
  if timeDim:
    f.createDimension('t',None)
  if nd > 2:
    f.createDimension('z',si[0])
    nd0 += 1
  f.createDimension('y',si[nd0])
  f.createDimension('x',si[nd0+1])

  alldims = ()
  if timeDim:
    tpo = f.createVariable('t', 'f', ('t',))
    alldims += ('t',)
  if nd > 2:
    zpo = f.createVariable('z', 'f', ('z',))
    alldims += ('z',)
  ypo = f.createVariable('y', 'f', ('y',))
  xpo = f.createVariable('x', 'f', ('x',))
  alldims += ('y', 'x',)

  varout = {}
  for key, psi in var.items():
    varout[key] = f.createVariable(key , 'f', alldims)

  if nd > 2:
    zpo[:] = np.arange(si[0])
  ypo[:] = np.arange(si[nd0])
  xpo[:] = np.arange(si[nd0+1])

  for key, psi in var.items():
    if timeDim:
      varout[key][0] = psi
      tpo[0] = 0
    else:
      varout[key][:] = psi

  f.close()


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
