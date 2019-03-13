#!/usr/bin/env python
# power_covariance.py - Linda Blot (linda.blot@obspm.fr) - 2013
# ---------------------------------- IMPORT ---------------------------------- #
from numpy import linalg
from power_stats import *
# ---------------------------------------------------------------------------- #


# -------------------------------- COVARIANCE POWER -------------------------------- #
def cov_power(powertype="power", mainpath="", simset=DeusPurSet("all_256"), isimmin=1, isimmax=2, noutput=1, aexp=0., nmodel=0, okprint=False, store=False, outpath=""):
    """ Covariance of power spectra. See power_spectrum for the description of the power spectrum types. If file exists it will be read from file.

    
    Parameters
    ---------
    powertype: string
        type of power spectrum: power, nyquist, renormalized, corrected, mcorrected, linear or linear_mock (default power)
    mainpath: string
        path to base folder (default empty)
    simset: Simset instance
        simulation set (default DeusPurSet("all_256"))
    isimmin: int
        mimimum simulation number (default 1)
    isimmax: int
        maximum simulation number (default 2)
    noutput: int
        snapshot number (default 1)
    aexp: float
        expansion factor (default 0)
    nmodel: int
        cosmological model number (default 0)
    okprint: bool
        verbose (default False)
    store: bool
        store file. If True and file exists it will be overwritten (default False)
    outpath: string
        path where output file is stored (default empty)
    
    Returns
    -------
    4 numpy arrays
        k, average power spectrum, standard deviation and covariance
    """

    nsim = isimmax-isimmin

    fname = outpath+"/"+output_file_name("cov", powertype, simset, isimmin, isimmax, noutput, nmodel)
    fltformat = "%-.12e"

    if os.path.isfile(fname):
        if okprint:
            print "Reading power spectrum covariance from file: ", fname
        power_k, power_pmean, power_psigma = mean_power(powertype, mainpath, simset, isimmin, isimmax, noutput, aexp, nmodel, okprint)
        power_pcov = np.loadtxt(fname, unpack=True)
    else:
        if okprint:
            print "Computing power spectrum covariance"
        if powertype == "linear":
            power_k, power_pmean = power_spectrum("linear", mainpath, simset, 1, noutput, aexp, nmodel, okprint)
            power_psigma = np.sqrt(2./simset.num_modes(power_k)) * power_pmean
            power_pcov = np.diag(power_psigma * power_psigma)
        else:
            power_k, power_pmean, power_psigma = mean_power(powertype, mainpath, simset, isimmin, isimmax, noutput, aexp, nmodel, okprint)
            power_pcov = np.zeros((power_k.size, power_k.size))
            for isim in xrange(isimmin, isimmax):
                if okprint:
                    true_simset, true_isim = sim_iterator(simset, isim)
                    print true_simset, true_isim
                dummy, power_p = power_spectrum(powertype, mainpath, true_simset, true_isim, noutput, aexp, nmodel, okprint)
                diff_power_p = power_p - power_pmean
                power_pcov += np.outer(diff_power_p, diff_power_p)
            power_pcov /= float(nsim-1)

        if store:
            f = open(fname, "w")
            for i in xrange(0, power_k.size):
                for j in xrange(0, power_k.size):
                    f.write(str(fltformat % power_pcov[i, j])+" ")
                f.write("\n")
            f.close()

    return power_k, power_pmean, power_psigma, power_pcov
# ---------------------------------------------------------------------------- #



# -------------------- CORRELATION COEFFICIENT POWER ------------------------- #
def corr_coeff(powertype="power", mainpath="", simset=DeusPurSet("all_256"), isimmin=1, isimmax=2, noutput=1,
               aexp=0., nmodel=0, okprint=False, store=False, outpath=""):
    """ Correlation coefficient of power spectra. See power_spectrum for the description of the power spectrum types. If file exists it will be read from file.

    
    Parameters
    ---------
    powertype: string
        type of power spectrum: power, nyquist, renormalized, corrected, mcorrected, linear or linear_mock (default power)
    mainpath: string
        path to base folder (default empty)
    simset: Simset instance
        simulation set (default DeusPurSet("all_256"))
    isimmin: int
        mimimum simulation number (default 1)
    isimmax: int
        maximum simulation number (default 2)
    noutput: int
        snapshot number (default 1)
    aexp: float
        expansion factor (default 0)
    nmodel: int
        cosmological model number (default 0)
    okprint: bool
        verbose (default False)
    store: bool
        store file. If True and file exists it will be overwritten (default False)
    outpath: string
        path where output file is stored (default empty)
    
    Returns
    -------
    4 numpy arrays
        k, average power spectrum, standard deviation and correlation coefficient matrix
    """

    fname = outpath+"/"+output_file_name("corr_coeff", powertype, simset, isimmin, isimmax, noutput, nmodel)
    fltformat = "%-.12e"
    if os.path.isfile(fname):
        if okprint:
            print "Reading power spectrum correlation coefficient from file: ", fname
        power_k, power_pmean, power_psigma = mean_power(powertype, mainpath, simset, isimmin, isimmax, noutput, aexp, nmodel, okprint)
        power_corr_coeff = np.loadtxt(fname, unpack=True)
    else:
        if okprint:
            print "Computing power spectrum correlation coefficient"
        power_k, power_pmean, power_psigma, power_pcov = cov_power(powertype, mainpath, simset, isimmin, isimmax, noutput, aexp, nmodel, okprint)
        norm = np.outer(power_psigma, power_psigma)
        power_corr_coeff = power_pcov / norm

        if store:
            f = open(fname, "w")
            for i in xrange(0, power_k.size):
                for j in xrange(0, power_k.size):
                    f.write(str(fltformat % power_corr_coeff[i, j])+" ")
                f.write("\n")
            f.close()

    return power_k, power_pmean, power_psigma, power_corr_coeff
# ---------------------------------------------------------------------------- #


# ----------------------------- SIGNAL TO NOISE ------------------------------ #
def signoise(powertype="nyquist", mainpath="", simset=DeusPurSet("all_256"), noutput=1, nsimmax=1, aexp=0., nmodel=0, unbiased=False, okprint=False, store=False, outpath=""):
    """ Signal to noise of power spectrum. See power_spectrum for the description of the power spectrum types. If file exists it will be read from file.

    
    Parameters
    ---------
    powertype: string
        type of power spectrum: power, nyquist, renormalized, corrected, mcorrected, linear or linear_mock (default power)
    mainpath: string
        path to base folder (default empty)
    simset: Simset instance
        simulation set (default DeusPurSet("all_256"))
    noutput: int
        snapshot number (default 1)
    nsimmax: int
        maximum number of simulations (default 1)
    aexp: float
        expansion factor (default 0)
    nmodel: int
        cosmological model number (default 0)
    unbiased: bool
        use Hartlap correction for inverse covariance (default False)
    okprint: bool
        verbose (default False)
    store: bool
        store file. If True and file exists it will be overwritten (default False)
    outpath: string
        path where output file is stored (default empty)
    
    Returns
    -------
    2 numpy arrays
        k_max and signal to noise
    """

    fname = outpath+"/"+output_file_name("sig_noise", powertype, simset, 1, nsimmax, noutput, nmodel)
    fltformat = "%-.12e"

    if os.path.isfile(fname):
        if okprint:
            print "Reading signal-to-noise from file: ", fname
        kmax, sig_noise = np.loadtxt(fname, unpack=True)
    else:
        if okprint:
            print "Computing signal-to-noise"
        nmax = nsimmax-1
        step = 1
        
        power_k, power_pmean, dummy, power_pcov = cov_power(powertype, mainpath, simset, 1, nsimmax, noutput, aexp, nmodel, okprint)

        num_iter = min(power_k.size, nmax-2)
        sig_noise = np.zeros(num_iter/step)
        kmax = np.zeros(num_iter/step)
        for ikmax in xrange(0, num_iter/step):
            ikk = ikmax*step+1
            kmax[ikmax] = power_k[ikk]
            idx = (power_k < kmax[ikmax])
            power_p_new = power_pmean[idx]
            power_pcov_kmax = power_pcov[0:ikk, 0:ikk]
            cov_inv = linalg.inv(power_pcov_kmax)
            if unbiased:
                cov_inv = float(nmax-num_iter-2) * cov_inv / float(nmax-1)
            sig_noise[ikmax] = math.sqrt(np.dot(power_p_new.T, np.dot(cov_inv, power_p_new)))

        if store:
            f = open(fname, "w")
            for i in xrange(0, kmax.size):
                f.write(str(fltformat % kmax[i])+" "+str(fltformat % sig_noise[i])+"\n")
            f.close()

    return kmax, sig_noise
# --------------------------------------------------------------------------- #
