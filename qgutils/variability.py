#!/usr/bin/env python

import numpy as np

def iterative_mean(it, psi_me, psi):
  '''

  Parameters
  ----------

  it : iteration number (starts at 1)
  psi_me : mean
  psi: field

  Returns
  -------

  psi_me : mean
  '''

  return psi_me + (psi - psi_me)/it


def iterative_variance(it, psi_me, psi_var, psi):
  '''
  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
  Welford's online algorithm

  Parameters
  ----------

  it : iteration number (starts at 1)
  psi_me : mean
  psi_var : variance
  psi: field

  Returns
  -------

  psi_me : mean
  psi_var : variance

  '''

  delta = psi - psi_me
  psi_me = psi_me + delta/it
  delta2 = psi - psi_me
  psi_var = (psi_var*(it-1) + delta*delta2)/it

  return psi_me, psi_var
