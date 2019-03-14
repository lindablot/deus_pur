#!/usr/bin/env python
# power_stats.py - Vincent Reverdy (vince.rev@gmail.com) and Linda Blot (linda.blot@obspm.fr) - 2013
# ---------------------------------- IMPORT ---------------------------------- #
from lmfit import minimize, Parameters
from power_types import *
# ---------------------------------------------------------------------------- #


# -------------------------------- MEAN POWER (ALTERNATIVE) -------------------------------- #
def mean_power(powertype="power", mainpath="", simset=DeusPurSet("all_256"), isimmin=1, isimmax=2, noutput=1, aexp=0.,
               nmodel=0, okprint=False, store=False,
               rebin=0, outpath=""):
    """ Mean and standard deviation of power spectra. See power_spectrum for the description of the power spectrum types. If file exists it will be read from file.

    
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
    rebin: int
        number of bins to combine when rebinning (default 0, i.e. no rebinning)
    outpath: string
        path where output file is stored (default empty)
    
    Returns
    -------
    3 numpy arrays
        k, average power spectrum and standard deviation
    """

    nsim = isimmax-isimmin
    fname = outpath+"/"+output_file_name("mean", powertype, simset, isimmin, isimmax, noutput, nmodel)

    if os.path.isfile(fname) and not store:
        if okprint:
            print "Reading mean spectrum from file: ", fname
        power_k, power_pmean, power_psigma = np.loadtxt(fname, unpack=True)
    else:
        if okprint:
            print "Computing mean and standard deviation of spectra"
        if powertype == "linear":
            power_k, power_pmean = power_spectrum("linear", mainpath, simset, 1, noutput,
                                                  aexp, nmodel, okprint, False, rebin)
            power_psigma = np.sqrt(2./simset.num_modes(power_k))*power_pmean
        else:
            power_k, dummy = power_spectrum(powertype, mainpath, simset, isimmin, noutput,
                                            aexp, nmodel, okprint, False, rebin)
            power_p = np.zeros((nsim, power_k.size))
            for isim in xrange(isimmin, isimmax):
                isim0 = isim - isimmin
                if okprint:
                    true_simset, true_isim = sim_iterator(simset, isim)
                    print true_simset, true_isim
                dummy, power_p[isim0] = power_spectrum(powertype, mainpath, simset, isim, noutput,
                                                       aexp, nmodel, okprint, False, rebin)

            power_pmean = np.mean(power_p, axis=0)
            if nsim > 1:
                power_psigma = np.std(power_p, axis=0, ddof=1)
            else:
                if okprint:
                    print "Standard deviation not computed because there is only one simulation"
                power_psigma = np.zeros(power_k.size)

        if store:
            if okprint:
                print "Writing file: ", fname
            f = open(fname, "w")
            for i in xrange(0, power_k.size):
                f.write(str("%-.12e" % power_k[i])+" "+str("%-.12e"%power_pmean[i])+" " +
                        str("%-.12e" % power_psigma[i])+"\n")
            f.close()

    return power_k, power_pmean, power_psigma
# ---------------------------------------------------------------------------- #



# --------------------- PDF OF POWER SPECTRA --------------------------- #
def distrib_power(powertype="power", mainpath="", simset=DeusPurSet("all_256"), isimmin=1, isimmax=2, noutput=1, nbin=50, kref=0.2,
                  aexp=0., nmodel=0, okprint=False, store=False, outpath=""):
    """ Distribution of power spectra at given k. See power_spectrum for the description of the power spectrum types. If file exists it will be read from file.

    
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
    nbin: int
        number of P(k) bins (default 50)
    kref: float
        k value (default 0.2 h/Mpc)
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
    2 numpy arrays
        P(k) bin center, frequency
    """

    nsim = isimmax - isimmin
    power_values = np.zeros(nsim)

    fname = outpath+"/"+output_file_name("distrib_k"+str(kref), powertype, simset, isimmin, isimmax, noutput, nmodel)
    if os.path.isfile(fname) and not store:
        bincenter, npower_bin = np.loadtxt(fname, unpack=True)
    else:
        for isim in xrange(1, nsim+1):
            if okprint:
                true_simset, true_isim = sim_iterator(simset, isim)
                print true_simset, true_isim
            power_k, power_p = power_spectrum(powertype, mainpath, simset, isim, noutput, aexp, nmodel)
            if kref > power_k[power_k.size-1] or kref < power_k[0]:
                raise ValueError("reference k value outside simulation k range in distrib_power")

            power_values[isim-1] = power_p[np.searchsorted(power_k, kref)]
        
        npower_bin, bins = np.histogram(power_values, nbin)
        npower_bin = np.asfarray(npower_bin)/float(nsim)
        bincenter = 0.5*(bins[1:]+bins[:-1])
        
        if store:
            if okprint:
                print "Writing file: ", fname
            f = open(fname, "w")
            for i in xrange(0, bincenter.size):
                f.write(str("%-.12e" % bincenter[i])+" "+str("%-.12e" % npower_bin[i])+"\n")
            f.close()

    return bincenter, npower_bin
# ---------------------------------------------------------------------------- #


# ------------------ HIGH MOMENTS OF SPECTRA PDF ---------------------- #
def high_moments(powertype="power", mainpath="", simset=DeusPurSet("all_256"), isimmin=1, isimmax=2, noutput=1,
                 aexp=0., nmodel=0, unbiased=True, okprint=False, store=False, outpath=""):
    """ Skewness and Kurtosis of the distribution of power spectra. See power_spectrum for the description of the power spectrum types. If file exists it will be read from file.

    
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
    unbiased: bool
        use unbiased estimators for Skewness and Kurtosis (default True)
    okprint: bool
        verbose (default False)
    store: bool
        store file. If True and file exists it will be overwritten (default False)
    outpath: string
        path where output file is stored (default empty)
    
    Returns
    -------
    3 numpy arrays
        k, skewness and kurtosis
    """

    nsim = isimmax-isimmin
    
    if unbiased:
        bias = "unbiased"
    else:
        bias = "biased"

    fname = outpath+"/"+output_file_name("high_moments_"+bias, powertype, simset, isimmin, isimmax, noutput, nmodel)
    if os.path.isfile(fname) and not store:
        power_k, power_skew, power_kurt = np.loadtxt(fname, unpack=True)
    else:
        power_k, power_pmean, power_psigma = mean_power(powertype, mainpath, simset, isimmin, isimmax, noutput,
                                                        aexp, nmodel, okprint, store)
        power_skew = np.zeros(power_k.size)
        power_kurt = np.zeros(power_k.size)
    
        for isim in xrange(isimmin, isimmax+1):
            if okprint:
                true_simset, true_isim = sim_iterator(simset, isim)
                print true_simset, true_isim
            power_k, power_p = power_spectrum(powertype, mainpath, simset, isim, noutput, aexp, nmodel)
            power_skew += (power_p-power_pmean)*(power_p-power_pmean)*(power_p-power_pmean)
            power_kurt += (power_p-power_pmean)*(power_p-power_pmean)*(power_p-power_pmean)*(power_p-power_pmean)

        if unbiased:
            power_skew *= float(nsim)/(float(nsim-1)*float(nsim-2))
            power_skew /= power_psigma * power_psigma * power_psigma
            power_kurt *= float(nsim+1) * float(nsim) / (float(nsim-1) * float(nsim-2) * float(nsim-3) *
                                                         power_psigma * power_psigma * power_psigma * power_psigma)
            power_kurt -= 3. * float(nsim-1) * float(nsim-1) / (float(nsim-2)*float(nsim-3))
        else:
            power_skew /= float(nsim)
            power_skew /= power_psigma * power_psigma * power_psigma
            power_kurt /= float(nsim)
            power_kurt /= power_psigma * power_psigma * power_psigma * power_psigma
            power_kurt -= 3.

        if store:
            if okprint:
                print "Writing file: ", fname
            f = open(fname, "w")
            for i in xrange(0, power_k.size):
                f.write(str("%-.12e" % power_k[i]) + " " + str("%-.12e" % power_skew[i]) + " " +
                        str("%-.12e" % power_kurt[i]) + "\n")
            f.close()
    
    return power_k, power_skew, power_kurt
# ---------------------------------------------------------------------------- #


# ------------------------- SPECTRUM CORRECTED FOR MASS RES EFFECT --------------------------- #
def mass_corrected_power(mainpath="", simset=DeusPurSet("all_256"), nsim=1, noutput=1,
                         aexp=0., corr_type="var_pres_smooth", okprint=False, store=False):
    """ Power spectrum corrected for mass resolution effects. See correction_power for different correction types. If file exists it is read from file.
    
    Parameters
    ---------
    mainpath: string
        path to base folder (default empty)
    simset: Simset instance
        simulation set (default DeusPurSet("all_256"))
    nsim: int
        simulation number (default 1)
    noutput: int
        snapshot number (default 1)
    aexp: float
        expansion factor (default 0)
    corr_type: string
        correction type: var_pres, var_pres_smooth, var_pres_pl, mean (default var_pres_smooth)
    okprint: bool
        verbose (default False)
    store: bool
        store file. If True and file exists it will be overwritten (default False)
        
    Returns
    -------
    2 numpy arrays
        k and P(k)
    """

    if simset.npart!=256 or (simset.name not in DeusPurSet.simsets):
        raise ValueError("mass resolution correction only valid for Deus Pur sets with npart=256")

    fname = input_file_name("power", mainpath, simset.name, nsim, noutput, 0, okprint, "mcorrected")
    if os.path.isfile(fname) and not store:
        power_k, corrected_p = np.loadtxt(fname, unpack=True)
    else:
        power_k, power_p = nyquist_power(mainpath, simset.name, nsim, noutput,
                                         aexp, growth_a, growth_dplus, okprint=okprint)
        correction_smooth = correction_power(mainpath, simset.name, noutput,
                                             aexp, corr_type, okprint=okprint, store=store)

        if corr_type == "var_pres" or corr_type == "var_pres_smooth" or corr_type == "var_pres_pl":
            simset_256 = DeusPurSet("all_256")
            power_k, power_pmean, power_psigma = mean_power("nyquist", mainpath, simset_256, 1,
                                                            simset_256.nsimmax+1, noutput,
                                                            aexp, okprint=okprint, store=store)
            simset_1024 = DeusPurSet("all_1024")
            power_k1024, power_pmean1024, power_psigma1024 = mean_power("nyquist", mainpath, simset_1024, 1,
                                                                        simset_1024.nsimmax+1, noutput, aexp, okprint=okprint, store=store)
            index = (power_k1024 <= power_k[power_k.size-1])
            power_pmean_1024 = power_pmean1024[index]
            corrected_p = correction_smooth * power_p + power_pmean_1024 - correction_smooth * power_pmean
        else:
            corrected_p = power_p / correction_smooth

        if store:
            if okprint:
                print "Writing file: ", fname
            f = open(fname, "w")
            for i in xrange(0, power_k.size):
                f.write(str("%-.12e" % power_k[i]) + " " + str("%-.12e" % corrected_p[i]) + "\n")
            f.close()

    return power_k, corrected_p
# ---------------------------------------------------------------------------- #


# ------------------------- CORRECTION TO THE LOW RES SPECTRA ------------------------------ #
def correction_power(mainpath="", simset=DeusPurSet("all_256"), noutput=1, aexp=1., corr_type="var_pres_smooth", okprint=False, store=False):
    """ Correction factor for mass resolution corrected power spectra.
    The correction types are:
    - var_pres: variance preserving correction
    - var_pres_smooth: variance preserving correction with 4th order polynomial smoothing
    - var_pres_pl: variance preserving correction with power law smoothing
    - mean: mean preserving correction
    
    Parameters
    ---------
    mainpath: string
        path to base folder (default empty)
    simset: Simset instance
        simulation set (default DeusPurSet("all_256"))
    noutput: int
        snapshot number (default 1)
    aexp: float
        expansion factor (default 0)
    corr_type: string
        correction type: var_pres, var_pres_smooth, var_pres_pl, mean (default var_pres_smooth)
    okprint: bool
        verbose (default False)
    store: bool
        store file. If True and file exists it will be overwritten (default False)
        
    Returns
    -------
    numpy array
        correction factor
    """

    fname = "correction_"+corr_type+"_"+str("%05d"%noutput)+".txt"
    if os.path.isfile(fname) and not store:
        correction = np.loadtxt(fname, unpack=True)
    else:
        simset_256 = DeusPurSet("all_256")
        simset_1024 = DeusPurSet("all_1024")

        if corr_type == "var_pres":
            if okprint:
                print "Mass resolution correction for mean and variance of the power spectrum with no smoothing."
            power_k, power_pmean, power_psigma = mean_power("nyquist", mainpath, simset_256, 1, simset_256.nsimmax+1,
                                                            noutput, aexp, okprint=okprint)
            power_k_CAMB, power_p_CAMB = read_power_camb(mainpath)
            growth_a, growth_dplus = read_growth(mainpath)
            aexp_end = 1.
            dplus_a = extrapolate([aexp], growth_a, growth_dplus)
            dplus_end = extrapolate([aexp_end], growth_a, growth_dplus)
            plin = power_p_CAMB * dplus_a * dplus_a / (dplus_end * dplus_end)
            plin_interp = np.interp(power_k, power_k_CAMB, plin)
            N_k = simset.num_modes(power_k)
            
            power_k1024, power_pmean1024, power_psigma1024 = mean_power("nyquist", mainpath, simset_1024, 1,
                                                                        simset_1024.nsimmax+1, noutput,
                                                                        aexp, okprint=okprint)
            index = (power_k1024 <= power_k[power_k.size-1])
            power_psigma_1024 = power_psigma1024[index]

            pvar_norm = power_psigma_1024 * power_psigma_1024 * N_k / (2. * plin_interp * plin_interp)
            fit_par = np.polyfit(power_k, pvar_norm, 9)
            fit = np.polyval(fit_par, power_k)
            
            correction = np.sqrt(fit * (2. * plin_interp * plin_interp) / N_k) / power_psigma

        elif corr_type == "var_pres_smooth":
            if okprint:
                print "Mass resolution correction for mean and variance of the power spectrum " \
                      "with polynomial smoothing (4th order)."
            power_k, power_pmean, power_psigma = mean_power("nyquist", mainpath, simset_256, 1, simset_256.nsimmax+1,
                                                            noutput, aexp, okprint=okprint)
            power_k1024, power_pmean1024, power_psigma1024 = mean_power("nyquist", mainpath, simset_1024, 1,
                                                                        simset_1024.nsimmax+1, noutput,
                                                                        aexp, okprint=okprint)
            index = (power_k1024 <= power_k[power_k.size-1])
            power_psigma_1024 = power_psigma1024[index]

            grade = 4
            fit_par = np.polyfit(power_k, power_psigma_1024/power_psigma, grade)
            correction = np.polyval(fit_par, power_k)

            params = Parameters()
            params.add('p0', value=1., vary=False)
            par = np.zeros(fit_par.size)
            par[0] = 1.
            for i in xrange(1, fit_par.size):
                params.add('p'+str("%01d" % i), value=fit_par[grade-i], vary=True)
                par[i] = fit_par[grade-i]
            poly = lambda p, x: params['p0'].value + params['p1'].value * x + params['p2'].value * x * x + \
                                params['p3'].value * x * x * x + params['p4'].value * x * x * x * x

            polyerr = lambda p, x, y: poly(p, x) - y

            fit = minimize(polyerr, params, args=(power_k, power_psigma_1024/power_psigma))
            for i in xrange(0, grade):
                par[i] = params['p'+str("%01d" % i)].value

            if okprint:
                print par

            x = power_k
            correction = params['p0'].value + params['p1'].value * x + params['p2'].value * x * x + \
                         params['p3'].value * x * x * x + params['p4'].value * x * x * x * x

        elif corr_type == "var_pres_pl":
            if okprint:
                print "Mass resolution correction for mean and variance of the power spectrum with power-law smoothing"
            power_k, power_pmean, power_psigma = mean_power("nyquist", mainpath, simset_256, 1, simset_256.nsimmax+1,
                                                            noutput, aexp, okprint=okprint)
            power_k1024, power_pmean1024, power_psigma1024 = mean_power("nyquist", mainpath, simset_1024, 1,
                                                                        simset_1024.nsimmax+1, noutput, aexp,
                                                                        okprint=okprint)
            index = (power_k1024 <= power_k[power_k.size-1])
            power_psigma_1024 = power_psigma1024[index]

            params = Parameters()
            params.add('alpha', value=1., vary=True)
            params.add('beta', value=2., vary=True)

            powlaw = lambda p, x: 1.+params['alpha'].value * x ** params['beta'].value
            
            powerr = lambda p, x, y: powlaw(p, x) - y

            fit = minimize(powerr, params, args=(power_k, power_psigma_1024/power_psigma))
            
            x = power_k
            correction = 1. + params['alpha'].value * x ** params['beta'].value

            if okprint:
                print 'alpha', 'beta'
                print params['alpha'].value, params['beta'].value
                print 'err_alpha', 'err_beta'
                print params['alpha'].stderr, params['beta'].stderr

        elif corr_type=="mean":
            if okprint:
                print "Mass resolution correction for mean power spectrum"
            power_k, power_pmean, power_psigma = mean_power("nyquist", mainpath, simset_256, 1, simset_256.nsimmax+1,
                                                            noutput, aexp, okprint=okprint)
            power_k1024, power_pmean1024, power_psigma1024 = mean_power("nyquist", mainpath, simset_1024, 1,
                                                                        simset_1024.nsimmax+1, noutput, aexp, okprint=okprint)
            index = (power_k1024 <= power_k[power_k.size-1])
            power_pmean_1024 = power_pmean1024[index]
            correction = power_pmean / power_pmean1024

        else:
            raise ValueError("unknown corr_type in correction_power")

        if store:
            if okprint:
                print "Writing file: ", fname
            f=open(fname, "w")
            for i in xrange(0, correction.size):
                f.write(str("%-.12e" % correction[i]) + "\n")
            f.close()
    
    return correction
# ---------------------------------------------------------------------------- #


# ----------------------- WRAPPER FOR ALL POWER TYPES ------------------------- #
def power_spectrum(powertype="power", mainpath="", simset=DeusPurSet("all_256"), nsim=1, noutput=1,
                   aexp=0., nmodel=0, okprint=False, store=False, rebin=0):
    """ Wrapper for different power spectrum types. If file exists it will be read from file.
    Different power types are:
    - power: raw power spectrum from file
    - renormalized: power spectrum renormalized to the initial power spectrum
    - corrected: power spectrum rescaled to given aexp using growth function
    - nyquist: power spectrum rescaled to given aexp using growth function and cut at half the nyquist frequency
    - mcorrected: power spectrum corrected for the mass resolution effect
    - linear: CAMB linear power spectrum rescaled to given aexp
    - linear_mock: CAMB linear power spectrum rescaled to given aexp + Gaussian error realisation
        
    Parameters
    ---------
    powertype: string
        type of power spectrum (power, nyquist, renormalized, corrected, mcorrected, linear, linear_mock)
    mainpath: string
        path to base folder
    simset: Simset instance
        simulation set (default DeusPurSet("all_256"))
    nsim: int
        simulation number
    noutput: int
        snapshot number
    aexp: float
        expansion factor (default 0)
    nmodel: int
        cosmological model number
    okprint: bool
        verbose (default False)
    store: bool
        store file. If True and file exists it will be overwritten (default False)
    rebin: int
        number of bins to combine when rebinning (default 0, i.e. no rebinning)
    
    Returns
    -------
    2 numpy arrays
        k and P(k)
    """

    aexp_info = read_aexp_info(input_file_name("info", mainpath, simset, nsim, noutput))
    if not np.isclose(aexp, aexp_info, atol=1.e-2):
        print "Warning: aexp is different from snapshot expansion factor by ", aexp-aexp_info
    setname, nsim = sim_iterator(simset, nsim)
    internal_simset = DeusPurSet(setname)
    if nmodel == 0:
        model = "lcdmw7"
    else:
        model = "model"+str(int(nmodel)).zfill(5)
    growth_a, growth_dplus = read_growth(mainpath, model)
    if powertype == "power":
        fname = input_file_name("power", mainpath, setname, nsim, noutput, nmodel)
        power_k, power_p, dummy = read_power_powergrid(fname)
    elif powertype == "nyquist":
        power_k, power_p = nyquist_power(mainpath, internal_simset, nsim, noutput,
                                         aexp, growth_a, growth_dplus, nmodel, okprint, store)
    elif powertype == "renormalized":
        power_k, power_p = renormalized_power(mainpath, internal_simset, nsim, noutput,
                                              growth_a, growth_dplus, nmodel, okprint, store)
    elif powertype == "corrected":
        power_k, power_p = corrected_power(mainpath, internal_simset, aexp, nsim, noutput,
                                           growth_a, growth_dplus, nmodel, okprint, store)
    elif powertype == "mcorrected":
        power_k, power_p = mass_corrected_power(mainpath, internal_simset, nsim, noutput,
                                                aexp, okprint=okprint, store=store)
    elif powertype == "linear":
        power_k_CAMB, power_p_CAMB = read_power_camb(mainpath, model)
        power_k_nocut, dummy, dummy = read_power_powergrid(fname)
        aexp_end = 1.
        dplus_a = extrapolate([aexp], growth_a, growth_dplus)
        dplus_end = extrapolate([aexp_end], growth_a, growth_dplus)
        plin = power_p_CAMB * dplus_a * dplus_a / (dplus_end * dplus_end)
        idx = (power_k_nocut < simset.nyquist)
        power_k = power_k_nocut[idx]
        power_p = np.interp(power_k, power_k_CAMB, plin)
        if store:
            linfname = input_file_name("power", mainpath, setname, nsim, noutput, nmodel, okprint, powertype)
            if okprint:
                print "Writing file: ", linfname
            f=open(linfname, "w")
            for i in xrange(0, power_k.size):
                f.write(str("%-.12e" % power_k[i])+" "+str("%-.12e" % power_p[i])+"\n")
            f.close()

    elif powertype == "linear_mock":
        if nmodel == 0:
            model = "lcdmw7"
        else:
            model = "model"+str(int(nmodel)).zfill(5)
        power_k_CAMB, power_p_CAMB = read_power_camb(mainpath, model)
        power_k, dummy, dummy = read_power_powergrid(input_file_name("power", mainpath, setname, nsim, noutput, nmodel))
        aexp_end = 1.
        dplus_a = extrapolate([aexp], growth_a, growth_dplus)
        dplus_end = extrapolate([aexp_end], growth_a, growth_dplus)
        plin = power_p_CAMB * dplus_a * dplus_a / (dplus_end * dplus_end)
                                                     
        plin_interp = np.interp(power_k, power_k_CAMB, plin)
        nmodes = simset.num_modes(power_k)
        noise = np.sqrt(2./nmodes)*plin_interp
        error = np.random.normal(0., noise, noise.size)
        power_p = plin_interp + error
    else:
        raise ValueError("powertype not existent in function power_spectrum")

    if rebin > 0:
        nmodes = simset.num_modes(power_k)
        power_k, power_p = rebin_pk(power_k, power_p, nmodes, rebin)

    return power_k, power_p
# ---------------------------------------------------------------------------- #
