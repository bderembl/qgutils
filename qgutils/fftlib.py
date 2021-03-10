#!/usr/bin/env python

import numpy as np

# basic spectral tools for data in square domains
# based on https://github.com/adekunleoajayi/powerspec

def azimuthal_integral(spec_2D, Delta):
  '''
  Compute the azimuthal integral of a 2D spectra

  Parameters
  ----------

  spec_2D: array [ny,nx] 2d spectra
  Delta: float, grid step

  Returns
  -------

  kr: array [nkr]  radial wave number
  spec_1D: array [nkr] 1d spectra
  '''

  N,naux = spec_2D.shape
  k,l,K,kr = get_wavenumber(N,Delta)
  dk = kr[0]
  spec_1D = np.zeros(len(kr))
  for i in range(kr.size):
    kfilt =  (K>=kr[i] - 0.5*dk) & (K<kr[i] + 0.5*dk)
    Nbin = kfilt.sum()
    # the azimuthal integral is the averge value*2*pi*k
    # but to get an equal integral for the 1d spetrum and 2d spectrum
    # it is better to just sum the cells*dk
    spec_1D[i] = (spec_2D[kfilt].sum())*dk #*kr[i]*2*np.pi/Nbin
  # the loop is missing the value at K=0:
  spec_1D[0] += spec_2D[int(N/2),int(N/2)]*dk
  return kr, spec_1D


def get_wavenumber(N, Delta, all_kr=False):
  '''
  Compute wavenumber and radial wavenumber 
  wave number units are in cycle per unit length

  Parameters
  ----------

  N : int
  Delta: float, grid step
  all_kr: bool. if False, only get radial wave number
    in the inner circle, if True, gets all wave numbers. 
    Default is False. You should only use True to check Parseval

  Returns
  -------

  k: array [ny,nx] zonal wave numbers
  l: array [ny,nx] meridional wave numbers
  K: array [ny,nx]  (k^2 + l^2)
  kr: array [nkr]  radial wave number
  '''

  kx = np.fft.fftshift(np.fft.fftfreq(N,Delta)) # two sided  
  k,l = np.meshgrid(kx,kx)
  K = np.sqrt(k**2 + l**2)
  if all_kr == False:
    kr = k[0,int(N/2)+1:]
  elif all_kr == True:
    kmax = K.max()
    dk = np.abs(kx[2]-kx[1])
    kr = dk*np.arange(1,int(kmax/dk)+1)
  return k,l,K,kr


def get_spec_2D(psi1, psi2, Delta, window=None):
  ''' 
  Compute the 2D cross power spectrum of psi1 and psi2
  If psi1=psi2, then this is simply the power spectrum

   normalization such that parseval is ok: 
   E = 1/V*(np.sum(psi**2)*Delta**2) = np.sum(psi**2)/N**2
     = np.sum(spec_2D)*dk**2
     ~ np.sum(spec_1D)*dk
  (the last equality is not exactly true because the 1d spectra only contains 
  the inner circle to the square k,l. Set all_kr=True in get_spec_1D)
  
  with V = (N*Delta)**2

  Parameters
  ----------

  psi1 : array [ny,nx]
  psi2 : array [ny,nx]
  Delta: grid step
  window: None or "hanning"

  Returns
  -------

  k: array [ny,nx] zonal wave numbers
  l: array [ny,nx] meridional wave numbers
  spec_2D: array [ny,nx] 2d spectra
  '''
  
  N,naux = psi1.shape
  k,l,K,kr = get_wavenumber(N,Delta)

  if window == 'hanning':
    win1d = np.hanning(N)
    window = win1d[None,:]*win1d[:,None]
  else:
    window = np.ones((N,N))

  psi1 = window*psi1
  psi2 = window*psi2

  psi1_hat = np.fft.fft2(psi1)
  psi2_hat = np.fft.fft2(psi2)
  spec_2D = (psi1_hat*psi2_hat.conj()).real*Delta**2/N**2
  spec_2D = np.fft.fftshift(spec_2D)
  return k, l, spec_2D


def get_spec_1D(psi1, psi2, Delta, window=None, all_kr= False):
  '''
  Compute the 1D power spectrum of the data by azimuthal integral
  of the 2D spectra

  Parameters
  ----------

  psi1 : array [ny,nx]
  psi2 : array [ny,nx]
  Delta: grid step
  window: None or "hanning"
  all_kr: bool. if False, only get radial wave number
    in the inner circle, if True, gets all wave numbers. 
    Default is False. You should only use True to check Parseval

  Returns
  -------

  kr: array [N] radial wave number
  spec_1D: array [N] 1d spectra

  '''

  k, l, spec_2D = get_spec_2D(psi1, psi2, Delta, window)
  kr, spec_1D = azimuthal_integral(spec_2D,Delta)
  return kr, spec_1D


def get_flux(psi1, psi2, Delta, window=None):
  '''
  Compute spectral flux

  Parameters
  ----------

  psi1 : array [ny,nx]
  psi2 : array [ny,nx]
  Delta: grid step
  window: None or "hanning"

  Returns
  -------

  kr: array [N] radial wave number
  flux: array [N] spectral flux

  '''
  k, l, spec_2D = get_spec_2D(psi1, psi2, Delta, window)

  # kr,spec_1D = azimuthal_integral(spec_2D,Delta)
  # flux = -np.cumsum(spec_1D)*dk # integrate from low wavenumbers
  # flux = np.cumsum(spec_1D[::-1])[::-1]*dk # integrate from high wavenumbers

  N,naux = psi1.shape
  k,l,K,kr = get_wavenumber(N,Delta)
  dk = kr[1] - kr[0]

  flux = np.zeros(len(kr))
  for i in range(kr.size):
    kfilt =  (kr[i] <= K ) 
    flux[i] = (spec_2D[kfilt]).sum()*dk*dk
  return kr, flux

