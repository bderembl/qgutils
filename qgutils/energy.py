#!/usr/bin/env python

import numpy as np

from .grid import *


def comp_vel(psi, Delta):

  '''
  Compute velocity at cell center

  u = -d psi /dy
  v =  d psi /dx

  '''

  psi_pad = pad_bc(psi)

  u = (psi_pad[:,:-2,1:-1] - psi_pad[:,2:,1:-1])/(2*Delta)
  v = (psi_pad[:,1:-1,2:] - psi_pad[:,1:-1,:-2])/(2*Delta)

  return u,v
