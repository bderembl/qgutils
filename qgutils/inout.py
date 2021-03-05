#!/usr/bin/env python

import numpy as np


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
