import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle
from astropy.io import fits
from scipy import stats
from sklearn.utils import check_random_state
import time as tm

import numpy as np


from scipy.fftpack import fft, ifft, fftshift


def find_nearest1(array,value):
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx


############
name = 'u18'
mjd = np.load('mjd'+name+'.npy')
mag = np.load('mag'+name+'.npy')
dmag = np.load('dmag'+name+'.npy')
obnames = np.load('obnames'+name+'.npy')
time = np.load('time'+name+'.npy')

maxdata = 0.7698648009572514

def FT_continuous(t, h, axis=-1, method=1):
    """Approximate a continuous 1D Fourier Transform with sampled data.

    This function uses the Fast Fourier Transform to approximate
    the continuous fourier transform of a sampled function, using
    the convention

    .. math::

       H(f) = \int h(t) exp(-2 \pi i f t) dt

    It returns f and H, which approximate H(f).

    Parameters
    ----------
    t : array_like
        regularly sampled array of times
        t is assumed to be regularly spaced, i.e.
        t = t0 + Dt * np.arange(N)
    h : array_like
        real or complex signal at each time
    axis : int
        axis along which to perform fourier transform.
        This axis must be the same length as t.

    Returns
    -------
    f : ndarray
        frequencies of result.  Units are the same as 1/t
    H : ndarray
        Fourier coefficients at each frequency.
    """
    assert t.ndim == 1
    assert h.shape[axis] == t.shape[0]
    N = len(t)
    if N % 2 != 0:
        raise ValueError("number of samples must be even")

    Dt = t[1] - t[0]
    Df = 1. / (N * Dt)
    t0 = t[N // 2]

    f = Df * (np.arange(N) - N // 2)

    shape = np.ones(h.ndim, dtype=int)
    shape[axis] = N

    phase = np.ones(N)
    phase[1::2] = -1
    phase = phase.reshape(shape)

    if method == 1:
        H = Dt * fft(h * phase, axis=axis)
    else:
        H = Dt * fftshift(fft(h, axis=axis), axes=axis)

    H *= phase
    H *= np.exp(-2j * np.pi * t0 * f.reshape(shape))
    H *= np.exp(-1j * np.pi * N / 2)

    return f, H



def PSD_continuous(t, h, axis=-1, method=1):
    """Approximate a continuous 1D Power Spectral Density of sampled data.

    This function uses the Fast Fourier Transform to approximate
    the continuous fourier transform of a sampled function, using
    the convention

    .. math::

        H(f) = \int h(t) \exp(-2 \pi i f t) dt

    It returns f and PSD, which approximate PSD(f) where

    .. math::

        PSD(f) = |H(f)|^2 + |H(-f)|^2

    Parameters
    ----------
    t : array_like
        regularly sampled array of times
        t is assumed to be regularly spaced, i.e.
        t = t0 + Dt * np.arange(N)
    h : array_like
        real or complex signal at each time
    axis : int
        axis along which to perform fourier transform.
        This axis must be the same length as t.

    Returns
    -------
    f : ndarray
        frequencies of result.  Units are the same as 1/t
    PSD : ndarray
        Fourier coefficients at each frequency.
    """
    assert t.ndim == 1
    assert h.shape[axis] == t.shape[0]
    N = len(t)
    if N % 2 != 0:
        raise ValueError("number of samples must be even")

    ax = axis % h.ndim

    if method == 1:
        # use FT_continuous
        f, Hf = FT_continuous(t, h, axis)
        Hf = np.rollaxis(Hf, ax)
        f = -f[N // 2::-1]
        PSD = abs(Hf[N // 2::-1]) ** 2
        PSD[:-1] += abs(Hf[N // 2:]) ** 2
        PSD = np.rollaxis(PSD, 0, ax + 1)
    else:
        # A faster way to do it is with fftshift
        # take advantage of the fact that phases go away
        Dt = t[1] - t[0]
        Df = 1. / (N * Dt)
        f = Df * np.arange(N // 2 + 1)
        Hf = fft(h, axis=axis)
        Hf = np.rollaxis(Hf, ax)
        PSD = abs(Hf[:N // 2 + 1]) ** 2
        PSD[-1] = 0
        PSD[1:] += abs(Hf[N // 2:][::-1]) ** 2
        PSD[0] *= 2
        PSD = Dt ** 2 * np.rollaxis(PSD, 0, ax + 1)

    return f, PSD





def generate_power_law(N, dt, beta, generate_complex=False,random_state=None):
    """Generate a power-law light curve
    This uses the method from Timmer & Koenig [1]_
    Parameters
    ----------
    N : integer
        Number of equal-spaced time steps to generate
    dt : float
        Spacing between time-steps
    beta : float
        Power-law index.  The spectrum will be (1 / f)^beta
    generate_complex : boolean (optional)
        if True, generate a complex time series rather than a real time series

    Returns
    -------
    x : ndarray
        the length-N
    References
    ----------
    .. [1] Timmer, J. & Koenig, M. On Generating Power Law Noise. A&A 300:707
    """
    random_state = check_random_state(random_state)
    dt = float(dt)
    N = int(N)

    Npos = int(N / 2)
    Nneg = int((N - 1) / 2)
    domega = (2 * np.pi / dt / N)

    if generate_complex:
        omega = domega * np.fft.ifftshift(np.arange(N) - int(N / 2))
    else:
        omega = domega * np.arange(Npos + 1)

    x_fft = np.zeros(len(omega), dtype=complex)
    x_fft.real[1:] = random_state.normal(0, 1, len(omega) - 1)
    x_fft.imag[1:] = random_state.normal(0, 1, len(omega) - 1)

    x_fft[1:] *= (1. / omega[1:]) ** (0.5 * beta)
    x_fft[1:] *= (1. / np.sqrt(2))

    # by symmetry, the Nyquist frequency is real if x is real
    if (not generate_complex) and (N % 2 == 0):
        x_fft.imag[-1] = 0

    if generate_complex:
        x = np.fft.ifft(x_fft)
    else:
        x = np.fft.irfft(x_fft, N)

    return x


def lcsim2(timestodo):
    start = tm.time()
    dt = 10./60./24. # ten minutes in days
    Nnumber = 30/dt # 21 days or 3 weeks
    beta = 2. # red noise beta = 2

    maxvals = []
    bestfreqs = []
    plsmax = []
    plots = False
    for i in np.arange(timestodo):



        ts = dt * np.arange(Nnumber)


        lcsimul = generate_power_law(Nnumber,dt,beta)


        #Add Noise
        noise =  np.random.normal(size=len(lcsimul),loc=np.mean(dmag),scale=np.std(dmag))
        lcsimulnoise = lcsimul + noise


        #Scale
        #lcsimulscalenoise = np.array([(i-lcsimulnoise.mean())/lcsimulnoise.std() for i in lcsimulnoise])
        lcsimulscalenoise = stats.zscore(lcsimulnoise)
        lcsimulscalenoise = lcsimulscalenoise*mag.std()+mag.mean()

        f, PSD = PSD_continuous(ts, lcsimulscalenoise)


        #
        newmjd = ts + mjd.min()
        mjdsort = np.sort(mjd)
        magsort = mag[np.argsort(mjd)]

        closestarg = np.array([np.abs(i-newmjd).argmin() for i in mjdsort])
        #closestarg = np.array([find_nearest1(newmjd,i) for i in mjdsort])



        simulmjdclose = newmjd[closestarg]
        lcsimulclose = lcsimulscalenoise[closestarg]
        noiseclose = noise[closestarg]


        freq, PLS = LombScargle(simulmjdclose, lcsimulclose, noiseclose).autopower(minimum_frequency=1 / 10.,
                                                        maximum_frequency=1 / 0.1,method='fast')
        best_freq = freq[np.argmax(PLS)]
        phase = (mjd * best_freq) % 1

        maxvals.append(PLS.max())
        bestfreqs.append(best_freq)
        if PLS.max() > maxdata:
            plsmax.append(PLS)
        if i % 1000 == 0:
            print(i)
            print("Taken", tm.time() - start, "seconds.")

        if plots:

            plt.rc('font', family='serif')
            plt.rc('xtick', labelsize='x-large')
            plt.rc('ytick', labelsize='x-large')

            fig = plt.figure(figsize=(20, 10))
            fig.subplots_adjust(wspace=0.1,hspace=0.5)

            # First axes: plot the time series
            ax1 = fig.add_subplot(211)
            ax1.title.set_text(f'Simulated Max P{PLS.max()} at {1/best_freq}')
            #ax1.set(xlim=(0.2, 10),
            #          ylim=(0, 1));
            ax1.set(ylim=(0,1));
            ax1.set_xlabel('Period (days)',fontsize=20)
            ax1.set_ylabel('Lomb-Scargle Power',fontsize=20)
            ax1.tick_params(axis='both', which='major', labelsize=28)


            ax1.plot(1./freq, PLS,color='k',ls='solid')

            #save iamge
            #fig.savefig('periodogram.eps', format='eps',bbox_inches = "tight")




            freq, PLS = LombScargle(mjd, mag, dmag).autopower(minimum_frequency=1 / 10.,
                                                            maximum_frequency=1 / 0.1)
            best_freq = freq[np.argmax(PLS)]
            phase = (mjd * best_freq) % 1


            # plot the periodogram



            plt.show()


