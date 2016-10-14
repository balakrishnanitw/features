# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 15:38:26 2016

@author: Balakrishna
"""

# -*- coding: utf-8 -*-
from numpy.fft import fft, ifft
#from numpy import isrealobj
#import tools
from numpy import real
from sklearn import preprocessing
import glob
#import LEVINSON


from scipy.io.wavfile import read
import numpy as np
#import matplotlib.pyplot as plt
wavs = []
#list_all = []
j_all = []

#from scipy.special import entr
for filename in glob.glob('*.wav'):
    #print(filename)
    wavs.append(read(filename))
    signal = read(filename)
    signal2array= np.array(signal[1], dtype=float)

__all__ = ['lpc']

def lpc(x, N):
    """Linear Predictor Coefficients.

    :param x:
    :param int N: default is length(X) - 1
    
    :Details:

    Finds the coefficients :math:`A=(1, a(2), \dots a(N+1))`, of an Nth order 
    forward linear predictor that predicts the current value value of the 
    real-valued time series x based on past samples:
    
    .. math:: \hat{x}(n) = -a(2)*x(n-1) - a(3)*x(n-2) - ... - a(N+1)*x(n-N)

    such that the sum of the squares of the errors

    .. math:: err(n) = X(n) - Xp(n)

    is minimized. This function  uses the Levinson-Durbin recursion to 
    solve the normal equations that arise from the least-squares formulation.  

    .. seealso:: :func:`levinson`, :func:`aryule`, :func:`prony`, :func:`stmcb`

    .. todo:: matrix case, references
    
    :Example:

    ::
    
        from scipy.signal import lfilter
        noise = randn(50000,1);  % Normalized white Gaussian noise
        x = filter([1], [1 1/2 1/3 1/4], noise)
        x = x[45904:50000]
        x.reshape(4096, 1)
    
        x = x[0]

    Compute the predictor coefficients, estimated signal, prediction error, and autocorrelation sequence of the prediction error:
   

    1.00000 + 0.00000i   0.51711 - 0.00000i   0.33908 - 0.00000i   0.24410 - 0.00000i

    ::
 
        a = lpc(x, 3)
        est_x = lfilter([0 -a(2:end)],1,x);    % Estimated signal
        e = x - est_x;                        % Prediction error
        [acs,lags] = xcorr(e,'coeff');   % ACS of prediction error

    
    a = lpc(signal2array, 3)
    """
    m = len(signal2array)    
#    if N == None:
#        N = m - 1 #default value if N is not provided
#    elif N > m-1:
#        #disp('Warning: zero-padding short input sequence')
#        signal2array.resize(N+1)
#        #todo: check this zero-padding. 

    X = fft(x, 2**nextpow2(2.*len(x)-1))
    R = real(ifft(abs(X)**2))
    R = R/(m-1.) #Biased autocorrelation estimate
    return levinson_1d(R, N)

def nextpow2(n):
    """Return the next power of 2 such as 2^p >= n.
    Notes
    -----
    Infinite and nan are left untouched, negative values are not allowed."""
    if np.any(n < 0):
        raise ValueError("n should be > 0")

    if np.isscalar(n):
        f, p = np.frexp(n)
        if f == 0.5:
            return p-1
        elif np.isfinite(f):
            return p
        else:
            return f
    else:
        f, p = np.frexp(n)
        res = f
        bet = np.isfinite(f)
        exa = (f == 0.5)
        res[bet] = p[bet]
        res[exa] = p[exa] - 1
        return res

def levinson_1d(r, order):
    """Levinson-Durbin recursion, to efficiently solve symmetric linear systems
    with toeplitz structure.

    Parameters
    ---------
    r : array-like
        input array to invert (since the matrix is symmetric Toeplitz, the
        corresponding pxp matrix is defined by p items only). Generally the
        autocorrelation of the signal for linear prediction coefficients
        estimation. The first item must be a non zero real.

    Levinson is a well-known algorithm to solve the Hermitian toeplitz
    equation:

                       _          _
        -R[1] = R[0]   R[1]   ... R[p-1]    a[1]
         :      :      :          :      *  :
         :      :      :          _      *  :
        -R[p] = R[p-1] R[p-2] ... R[0]      a[p]
                       _
    with respect to a (  is the complex conjugate). Using the special symmetry
    in the matrix, the inversion can be done in O(p^2) instead of O(p^3).
    """
    r = np.atleast_1d(r)
    if r.ndim > 1:
        raise ValueError("Only rank 1 are supported for now.")

    n = r.size
    if n < 1:
        raise ValueError("Cannot operate on empty array !")
    elif order > n - 1:
        raise ValueError("Order should be <= size-1")

   # if not np.isreal(r[0]):
    ##    raise ValueError("First item of input must be real.")
    #elif not np.isfinite(1/r[0]):
     #   raise ValueError("First item should be != 0")

    # Estimated coefficients
    a = np.empty(order+1, r.dtype)
    # temporary array
    t = np.empty(order+1, r.dtype)
    # Reflection coefficients
    k = np.empty(order, r.dtype)

    a[0] = 1.
    e = r[0]

    for i in xrange(1, order+1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i-j]
        k[i-1] = -acc / e
        a[i] = k[i-1]

        for j in range(order):
            t[j] = a[j]
           # print a
        for j in range(1, i):
            a[j] += k[i-1] * np.conj(t[i-j])

        e *= 1 - k[i-1] * np.conj(k[i-1])

    return a, e, k

#LPCcoeff,Err,Ka = lpc(signal2array,20)
#list_all.append(LPCcoeff)
#j=LPCcoeff.tolist()
#for filename in glob.glob('*.wav'):
   # j_all.append(j)
for filename in glob.glob('*.wav'):
    #print(filename)
     wavs.append(read(filename))
     signal = read(filename)
     signal2array= np.array(signal[1], dtype=float)
     LPCcoeff,Err,Ka = lpc(signal2array,11)
     j=LPCcoeff.tolist()
     j_all.append(LPCcoeff)
lpc_scaled= preprocessing.scale(j_all)
lpc_reshaped=lpc_scaled.reshape(1,-1)
lpc_normalized = preprocessing.normalize(lpc_reshaped,norm='l2')
lpc_normalzed_cv = lpc_normalized.T
     