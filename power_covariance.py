#!/usr/bin/env python
# power_covariance.py - Linda Blot (linda.blot@obspm.fr) - 2013
# ---------------------------------- IMPORT ---------------------------------- #
from numpy import linalg
from power_stats import *
# ---------------------------------------------------------------------------- #


# -------------------------------- COVARIANCE POWER -------------------------------- #
def cov_power(powertype="power", mainpath="", simset=DeusPurSet("all_256"), isimmin=1, isimmax=2, noutput=1, aexp=0., okprint=False, store=False, rebin=0, irsd=0, multipole=0, shotnoise=True, mask=0, outpath="."):
    """ Covariance of power spectra. See power_spectrum for the description of the power spectrum types. If file exists it will be read from file.


    Parameters
    ---------
    powertype: string
        type of power spectrum: power, nyquist, renormalized, corrected, mcorrected, linear, linear_mock or [sample_name] (default power)
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
    okprint: bool
        verbose (default False)
    store: bool
        store file. If True and file exists it will be overwritten (default False)
    rebin: int
        number of bins to combine when rebinning (default 0, i.e. no rebinning)
    irsd: int
        direction of redshift space index (0 for real space, 1-3 for x,y or z, default 0)
    multipole: int
        multipole index (default 0)
    shotnoise: bool
        correct for Poisson shotnoise (default True)
    mask: int
        mask index (default 0)
    outpath: string
        path where output file is stored (default .)

    Returns
    -------
    4 numpy arrays
        k, average power spectrum, standard deviation and covariance
    """

    nsim = isimmax-isimmin

    fname = outpath+"/"+output_file_name("cov", powertype, simset, isimmin, isimmax, noutput, multipole, irsd, mask)
    fltformat = "%-.12e"

    if os.path.isfile(fname):
        if okprint:
            print "Reading power spectrum covariance from file: ", fname
        power_k, power_pmean, power_psigma = mean_power(powertype, mainpath, simset, isimmin, isimmax, noutput, aexp, okprint=okprint, store=False, rebin=rebin, irsd=irsd, multipole=multipole, shotnoise=shotnoise, mask=mask, outpath = outpath)
        power_pcov = pd.read_csv(fname, delim_whitespace=True, header=None).values
    else:
        if okprint:
            print "Computing power spectrum covariance"
        if powertype == "linear":
            power_k, power_pmean = power_spectrum("linear", mainpath, simset, 1, noutput, aexp, okprint, rebin = rebin)
            power_psigma = np.sqrt(2./simset.num_modes(power_k)) * power_pmean
            power_pcov = np.diag(power_psigma * power_psigma)
        else:
            power_k, power_p = load_power(powertype, mainpath, simset, noutput, aexp, rebin = rebin, outpath = outpath)
            power_p = power_p[isimmin-1:isimmax]
            power_k, power_pmean, power_psigma = mean_power(powertype, mainpath, simset, isimmin, isimmax, noutput, aexp, okprint, False, rebin, irsd, multipole, shotnoise, mask, outpath = outpath)
            power_pcov=np.cov(power_p,rowvar=False)

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
               aexp=0., okprint=False, store=False, rebin=0, irsd=0, multipole=0, shotnoise=True, mask=0, outpath="."):
    """ Correlation coefficient of power spectra. See power_spectrum for the description of the power spectrum types. If file exists it will be read from file.


    Parameters
    ---------
    powertype: string
        type of power spectrum: power, nyquist, renormalized, corrected, mcorrected, linear, linear_mock or [sample_name] (default power)
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
    okprint: bool
        verbose (default False)
    store: bool
        store file. If True and file exists it will be overwritten (default False)
    rebin: int
        number of bins to combine when rebinning (default 0, i.e. no rebinning)
    irsd: int
        direction of redshift space index (0 for real space, 1-3 for x,y or z, default 0)
    multipole: int
        multipole index (default 0)
    shotnoise: bool
        correct for Poisson shotnoise (default True)
    mask: int
        mask index (default 0)
    outpath: string
        path where output file is stored (default .)

    Returns
    -------
    4 numpy arrays
        k, average power spectrum, standard deviation and correlation coefficient matrix
    """

    fname = outpath+"/"+output_file_name("corr_coeff", powertype, simset, isimmin, isimmax, noutput, multipole, irsd, mask)
    fltformat = "%-.12e"
    if os.path.isfile(fname):
        if okprint:
            print "Reading power spectrum correlation coefficient from file: ", fname
        power_k, power_pmean, power_psigma = mean_power(powertype, mainpath, simset, isimmin, isimmax, noutput, aexp, okprint, False, rebin, irsd, multipole, shotnoise, mask, outpath = outpath)
        power_corr_coeff = pd.read_csv(fname, delim_whitespace=True, header=None).values
    else:
        if okprint:
            print "Computing power spectrum correlation coefficient"
        power_k, power_pmean, power_psigma, power_pcov = cov_power(powertype, mainpath, simset, isimmin, isimmax, noutput, aexp, okprint, False, rebin, irsd, multipole, shotnoise, mask, outpath = outpath)
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



# ---------------------- MULTIPOLE CROSS COVARIANCE -------------------------- #
def cross_mpole_cov_power(powertype = "power", mainpath = "", simset = MinervaSet("Minerva"), isimmin = 1, isimmax = 2, noutput = 1, aexp = 0., nmodel = 0, okprint = False, store = False, rebin = 0, irsd=0, multipole1 = 0, multipole2 = 1, shotnoise = True, mask = 0, outpath = "."):
    """ Cross covariance of two multipoles of the power spectrum. See power_spectrum for the description of the power spectrum types. If file exists it will be read from file.


    Parameters
    ---------
    powertype: string
        type of power spectrum: power, nyquist, renormalized, corrected, mcorrected, linear, linear_mock or [sample_name] (default power)
    mainpath: string
        path to base folder (default empty)
    simset: Simset instance
        simulation set (default MinervaSet("Minerva"))
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
    rebin: int
        number of bins to combine when rebinning (default 0, i.e. no rebinning)
    irsd: int
        direction of redshift space index (0 for real space, 1-3 for x,y or z, default 0)
    multipole1: int
        first multipole index (default 0)
    multipole2: int
        second multipole index (default 1)
    shotnoise: bool
        correct for Poisson shotnoise (default True)
    mask: int
        mask index (default 0)
    outpath: string
        path where output file is stored (default .)

    Returns
    -------
    2 numpy arrays
        k and covariance
    """

    nsim = isimmax-isimmin

    fname = outpath + "/" + output_file_name("cross_cov", powertype, simset, isimmin, isimmax, noutput, multipole1, irsd, mask, multipole2)
    fltformat="%-.12e"

    if (os.path.isfile(fname) and not store):
        if (okprint):
            print "Reading power spectrum covariance from file: ",fname
        power_k, dummy, dummy = power_spectrum(powertype, mainpath, simset, isimmin, noutput, aexp, nmodel, okprint, False, rebin, irsd, multipole1, shotnoise, mask)
        power_pcov = np.loadtxt(fname,unpack=True)
    else:
        if (okprint):
            print "Computing power spectrum covariance"

        power_k, power_pmean1, dummy = mean_power(powertype, mainpath, simset, isimmin, isimmax, noutput, aexp, nmodel, okprint, False, rebin, irsd, multipole1, shotnoise, mask, outpath)
        power_k, power_pmean2, dummy = mean_power(powertype, mainpath, simset, isimmin, isimmax, noutput, aexp, nmodel, okprint, False, rebin, irsd, multipole2, shotnoise, mask, outpath)
        diff_power_p1 = np.zeros(power_k.size)
        diff_power_p2 = np.zeros(power_k.size)
        power_pcov = np.zeros((power_k.size,power_k.size))
        for isim in xrange(isimmin, isimmax):
            dummy, power_p1 = power_spectrum(powertype, mainpath, simset, isim, noutput, aexp, nmodel, okprint, False, rebin, irsd, multipole1, shotnoise, mask)
            dummy, power_p2 = power_spectrum(powertype, mainpath, simset, isim, noutput, aexp, nmodel, okprint, False, rebin, irsd, multipole2, shotnoise, mask)
            diff_power_p1 = power_p1 - power_pmean1
            diff_power_p2 = power_p2 - power_pmean2
            power_pcov += np.outer(diff_power_p1,diff_power_p2)
        power_pcov/=float(nsim-1)

        if (store):
            f = open(fname, "w")
            for i in xrange(0, power_k.size):
                for j in xrange(0, power_k.size):
                    f.write(str(fltformat%power_pcov[i,j])+" ")
                f.write("\n")
            f.close()

    return power_k, power_pcov
# ---------------------------------------------------------------------------- #



# -------------------- FULL MULTIPOLES COVARIANCE ---------------------------- #
def mpole_cov_power(powertype = "power", mainpath = "", simset = "", isimmin = 1, isimmax = 2, noutput = 1, aexp = 0., nmodel = 0, okprint = False, store = False, rebin = 0, irsd = 0, shotnoise = True, mask = 0, outpath="."):
    """ Full covariance of the three multipoles of the power spectrum. See power_spectrum for the description of the power spectrum types. If file exists it will be read from file.


    Parameters
    ---------
    powertype: string
        type of power spectrum: power, nyquist, renormalized, corrected, mcorrected, linear, linear_mock or [sample_name] (default power)
    mainpath: string
        path to base folder (default empty)
    simset: Simset instance
        simulation set (default MinervaSet("Minerva"))
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
    rebin: int
        number of bins to combine when rebinning (default 0, i.e. no rebinning)
    irsd: int
        direction of redshift space index (0 for real space, 1-3 for x,y or z, default 0)
    shotnoise: bool
        correct for Poisson shotnoise (default True)
    mask: int
        mask index (default 0)
    outpath: string
        path where output file is stored (default .)

    Returns
    -------
    2 numpy arrays
        k and covariance
    """

    cov_file = outpath + "/" + output_file_name("full_cov", powertype, simset, isimmin, isimmax, noutput, -1, irsd, mask)
    if os.path.isfile(cov_file) and not store:
        if okprint:
            print "Reading full covariance from file ",cov_file
        full_cov = np.loadtxt(cov_file,unpack=True)
        k11, dummy = power_spectrum(powertype, mainpath, simset, isimmin, noutput, aexp, nmodel, False, False, rebin, 1, 1, shotnoise, mask)
    else:
        if okprint:
            "Computing covariance"
        k11, mean11, sigma11, cov11 = cov_power(powertype, mainpath, simset, isimmin, isimmax, noutput, aexp, nmodel, okprint, False, rebin, irsd, 1, shotnoise, mask, outpath)
        k22, mean22, sigma22, cov22 = cov_power(powertype, mainpath, simset, isimmin, isimmax, noutput, aexp, nmodel, okprint, False, rebin, irsd, 2, shotnoise, mask, outpath)
        k33, mean33, sigma33, cov33 = cov_power(powertype, mainpath, simset, isimmin, isimmax, noutput, aexp, nmodel, okprint, False, rebin, irsd, 3, shotnoise, mask, outpath)

        k12, cov12 = cross_mpole_cov_power(powertype, mainpath, simset, isimmin, isimmax, noutput, aexp, nmodel, okprint, False, rebin, irsd, 1, 2, shotnoise, mask, outpath)
        k13, cov13 = cross_mpole_cov_power(powertype, mainpath, simset, isimmin, isimmax, noutput, aexp, nmodel, okprint, False, rebin, irsd, 1, 3, shotnoise, mask, outpath)
        k23, cov23 = cross_mpole_cov_power(powertype, mainpath, simset, isimmin, isimmax, noutput, aexp, nmodel, okprint, False, rebin, irsd, 2, 3, shotnoise, mask, outpath)

        full_cov1 = np.concatenate((cov11,cov12,cov13), axis=1)
        full_cov2 = np.concatenate((cov12.T,cov22,cov23), axis=1)
        full_cov3 = np.concatenate((cov13.T,cov23.T,cov33), axis=1)
        full_cov = np.concatenate((full_cov1,full_cov2,full_cov3), axis=0)

        nbin = k11.size
        if (store):
            if okprint:
                print "Storing covariance in file ", cov_file
            f = open(cov_file, "w")
            for i in xrange(0, nbin):
                for j in xrange(0, nbin):
                    f.write(str("%-.12e"%cov11[i,j])+" ")
                for j in xrange(0, nbin):
                    f.write(str("%-.12e"%cov12[i,j])+" ")
                for j in xrange(0, nbin):
                    f.write(str("%-.12e"%cov13[i,j])+" ")
                f.write("\n")
            for i in xrange(0, nbin):
                for j in xrange(0, nbin):
                    f.write(str("%-.12e"%cov12[j,i])+" ")
                for j in xrange(0, nbin):
                    f.write(str("%-.12e"%cov22[i,j])+" ")
                for j in xrange(0, nbin):
                    f.write(str("%-.12e"%cov23[i,j])+" ")
                f.write("\n")
            for i in xrange(0, nbin):
                for j in xrange(0, nbin):
                    f.write(str("%-.12e"%cov13[j,i])+" ")
                for j in xrange(0, nbin):
                    f.write(str("%-.12e"%cov23[j,i])+" ")
                for j in xrange(0, nbin):
                    f.write(str("%-.12e"%cov33[i,j])+" ")
                f.write("\n")
            f.close()

    return k11, full_cov
# ---------------------------------------------------------------------------- #



# ----------------------------- SIGNAL TO NOISE ------------------------------ #
def signoise(powertype="nyquist", mainpath="", simset=DeusPurSet("all_256"), noutput=1, nsimmax=1, aexp=0., unbiased=False, okprint=False, store=False, rebin=0, outpath="."):
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
    unbiased: bool
        use Hartlap correction for inverse covariance (default False)
    okprint: bool
        verbose (default False)
    store: bool
        store file. If True and file exists it will be overwritten (default False)
    rebin: int
        number of bins to combine when rebinning (default 0, i.e. no rebinning)
    outpath: string
        path where output file is stored (default .)

    Returns
    -------
    2 numpy arrays
        k_max and signal to noise
    """

    fname = outpath+"/"+output_file_name("sig_noise", powertype, simset, 1, nsimmax, noutput)
    fltformat = "%-.12e"

    if os.path.isfile(fname):
        if okprint:
            print "Reading signal-to-noise from file: ", fname
        file_content = pd.read_csv(fname, delim_whitespace=True, header=None).values.T
        kmax = file_content[0]
        sig_noise = file_content[1]
    else:
        if okprint:
            print "Computing signal-to-noise"
        nmax = nsimmax-1
        step = 1

        power_k, power_pmean, dummy, power_pcov = cov_power(powertype, mainpath, simset, 1, nsimmax, noutput, aexp, okprint, rebin = rebin, outpath = outpath)

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
