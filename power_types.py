#!/usr/bin/env python
# power_types.py - Vincent Reverdy (vince.rev@gmail.com) and Linda Blot (linda.blot@obspm.fr) - 2013
# ---------------------------------- IMPORT ---------------------------------- #
from read_files import *
from utilities import *
# ---------------------------------------------------------------------------- #


# --------------------------- RENORMALIZED POWER SPECTRUM ----------------------------- #
def renormalized_power(mainpath="", simset=DeusPurSet("all_256"), nsim=1, noutput=1, growth_a=np.zeros(0),
                       growth_dplus=np.zeros(0), nmodel=0, okprint=False, store=False):
    """ Power spectrum renormalised to initial power spectrum using growth function. If file exists it will be read from file
    
    Parameters
    ----------
    mainpath: string
        path to base folder
    simset: Simset instance
        simulation set (default DeusPurSet("all_256"))
    nsim: int
        simulation number (default 1)
    noutput: int
        snapshot number (default 1)
    growth_a: numpy array
        expansion factor for growth function (default 0)
    growth_dplus: numpy array
        growth function at expansion factors growth_a (default 0)
    nmodel: int
        number of cosmological model (default 0)
    okprint: bool
        verbose (default False)
    store: bool
        store file. If True and file exists it will be overwritten (default False)
        
    Returns
    -------
    2 numpy arrays
        k and P(k)
    """

    fname = input_file_name("power", mainpath, simset, nsim, noutput, nmodel, okprint, "renormalized")
    if os.path.isfile(fname) and not store:
        file_content = pd.read_csv(fname, " ", header=None).values
        power_k = file_content[0]
        power_p = file_content[1]
    else:
        power_k, power_p_ini, dummy = read_power_powergrid(input_file_name("power", mainpath, simset, nsim, 1, nmodel))
        power_k, power_p_end, dummy = read_power_powergrid(input_file_name("power", mainpath, simset, nsim, noutput, nmodel))
        aexp_ini = read_aexp_info(input_file_name("info", mainpath, simset, nsim, 1, nmodel))
        aexp_end = read_aexp_info(input_file_name("info", mainpath, simset, nsim, noutput, nmodel))
        dplus_ini = extrapolate([aexp_ini], growth_a, growth_dplus)
        dplus_end = extrapolate([aexp_end], growth_a, growth_dplus)
        power_p = (power_p_end * dplus_ini * dplus_ini) / (power_p_ini * dplus_end * dplus_end)
        if store:
            if okprint:
                print "Writing file: ", fname
            f = open(fname, "w")
            for i in xrange(0, power_k.size):
                f.write(str("%-.12e" % power_k[i])+" "+str("%-.12e" % power_p[i]) + "\n")
            f.close()
    return power_k, power_p
# ---------------------------------------------------------------------------- #


# ----------------------------- POWER SPECTRUM CORRECTED FOR a DIFFERENCES ------------------------------ #
def corrected_power(mainpath="", simset=DeusPurSet("all_256"), nsim=1, noutput=1, aexp=0., growth_a=np.zeros(0),
                    growth_dplus=np.zeros(0), nmodel=0, okprint=False, store=False):
    """ Power spectrum rescaled to given aexp using growth function. If file exists it will be read from file
    
    Parameters
    ----------
    mainpath: string
        path to base folder
    simset: Simset instance
        simulation set (default DeusPurSet("all_256"))
    nsim: int
        simulation number (default 1)
    noutput: int
        snapshot number (default 1)
    aexp: float
        expansion factor (default 0)
    growth_a: numpy array
        expansion factor (default 0)
    growth_dplus: numpy array
        growth function at expansion factors growth_a (default 0)
    nmodel: int
        number of cosmological model (default 0)
    okprint: bool
        verbose (default False)
    store: bool
        store file. If True and file exists it will be overwritten (default False)
        
    Returns
    -------
    2 numpy arrays
        k and P(k)
    """

    fname = input_file_name("power", mainpath, simset, nsim, noutput, nmodel, okprint, "corrected")
    if os.path.isfile(fname) and not store:
        file_content = pd.read_csv(fname, " ", header=None).values
        power_k = file_content[0]
        power_p = file_content[1]
    else:
        power_k, power_p_raw, dummy = read_power_powergrid(input_file_name("power", mainpath, simset, nsim, noutput, nmodel))
        if aexp != 0.:
            aexp_raw = read_aexp_info(input_file_name("info", mainpath, simset, nsim, noutput, nmodel))
            dplus_raw = extrapolate([aexp_raw], growth_a, growth_dplus)
            dplus = extrapolate([aexp], growth_a, growth_dplus)
            power_p = (power_p_raw * dplus * dplus) / (dplus_raw * dplus_raw)
        else:
            power_p = power_p_raw
        if store:
            if okprint:
                print "Writing file: ", fname
            f = open(fname, "w")
            for i in xrange(0, power_k.size):
                f.write(str("%-.12e" % power_k[i]) + " " + str("%-.12e" % power_p[i]) + "\n")
            f.close()
    return power_k, power_p
# ---------------------------------------------------------------------------- #


# ----------------------------- POWER SPECTRUM CUT AT NYQUIST FREQUENCY ------------------------------ #
def nyquist_power(mainpath="", simset=DeusPurSet("all_256"), nsim=1, noutput=1, aexp=0., growth_a=np.zeros(0),
                  growth_dplus=np.zeros(0), nmodel=0, okprint=False, store=False):
    """ Power spectrum rescaled to given aexp using growth function and cut at half the nyquist frequency. If file exists it will be read from file
        
    Parameters
    ----------
    mainpath: string
        path to base folder
    simset: Simset instance
        simulation set (default DeusPurSet("all_256"))
    nsim: int
        simulation number (default 1)
    noutput: int
        snapshot number (default 1)
    aexp: float
        expansion factor (default 0)
    growth_a: numpy array
        expansion factorfor grwoth function (default 0)
    growth_dplus: numpy array
        growth function at expansion factors growth_a (default 0)
    nmodel: int
        number of cosmological model (default 0)
    okprint: bool
        verbose (default False)
    store: bool
        store file. If True and file exists it will be overwritten (default False)
        
    Returns
    -------
    2 numpy arrays
        k and P(k)
    """

    fname = input_file_name("power", mainpath, simset, nsim, noutput, nmodel, okprint, "nyquist")
    if os.path.isfile(fname) and not store:
        file_content = pd.read_csv(fname, " ", header=None).values
        power_k_new = file_content[0]
        power_p_new = file_content[1]
    else:
        power_k, power_p = corrected_power(mainpath, simset, nsim, noutput, aexp, growth_a, growth_dplus, nmodel)
        idx = (power_k < simset.nyquist)
        power_k_new = power_k[idx]
        power_p_new = power_p[idx]
        if store:
            if okprint:
                print "Writing file: ", fname
            f = open(fname, "w")
            for i in xrange(0, power_k_new.size):
                f.write(str("%-.12e" % power_k_new[i])+" "+str("%-.12e" % power_p_new[i]) + "\n")
            f.close()
    return power_k_new, power_p_new
# ---------------------------------------------------------------------------- #


# ------------------------------- POWER SPECTRUM COMPUTED BY PkANN ------------------ #
def pkann_power(par, redshift, power_k=np.zeros(0), okprint=False, store=False):
    """ Power spectrum computed by PkANN emulator
    
    Parameters
    ----------
    par: list
        cosmological parameters omega_m h^2, omega_b h^2, n_s, w, sigma_8, m_nu, bias
    redshift: float
        redshift
    power_k: numpy array
        array of k values where the power spectrum will be interpolated (optional)
    okprint: bool
        verbose (default False)
    store: bool
        store file. If True and file exists it will be overwritten (default False)
        
    Returns
    -------
    2 numpy arrays
        k and P(k)
    """

    folder = "pkann_spectra/"
    command = "mkdir -p " + folder
    os.system(command)

    pfile = folder+"pkann_"+str(ipar)+"_"+str(dpar)+"_"+str(fact)+"_"+powertype+"_"+str(ioutput)+".txt"
    if os.path.isfile(pfile) and not store:
        if okprint:
            print "Reading file ", pfile
        file_content = pd.read_csv(pfile, " ", header=None).values
        power_k = file_content[0]
        power_pkann = file_content[1]
    else:
        if okprint:
            print "Running Pkann"
        pkanndir = "/data/lblot/deus_pur/PkANN/"
        infile1 = pkanndir+"Testing_set/parameters.txt"
        infile2 = pkanndir+"Testing_set/z_list.txt"
        outfile = pkanndir+"Testing_set/PkANN_predictions/model_1/ps_nl_1.dat"
        fltformat = "%1.8f"
        f1 = open(infile1, "w")
        f1.write("   " + str(fltformat % par[0]) + "   "+str(fltformat % par[1]) + "   "+str(fltformat % par[2])+"   " +
                 str(fltformat % par[3])+"   "+str(fltformat % par[4])+"   "+str(fltformat % par[5])+"\n")
        f1.close()
        f2 = open(infile2, "w")
        f2.write(str(fltformat % redshift)+"\n")
        f2.close()

        command = "cd "+pkanndir+"; ./unix.sh > output.out; cd - > tmp"
        os.system(command)
        file_content = pd.read_csv(outfile, " ", header=None, skiprows=4).values
        kpkann = file_content[0]
        ppkann = file_content[1]

        if power_k.size > 0:
            power_pkann = par[6]*par[6]*np.interp(power_k,kpkann,ppkann)
        else:
            power_k = kpkann
            power_pkann = par[6]*par[6]*ppkann

        if store:
            fltformat2 = "%-.12e"
            fout = open(pfile, "w")
            for i in xrange(0, power_k.size):
                fout.write(str(fltformat2 % power_k[i])+" "+str(fltformat2 % power_pkann[i])+"\n")
            fout.close()

    return power_k, power_pkann
# ---------------------------------------------------------------------------- #


# --------------- POWER SPECTRUM COMPUTED BY EuclidEmulator ------------------ #
def euemu_power(par, redshift, power_k=np.zeros(0)):
    """ Power spectrum computed by Euclid emulator
    
    Parameters
    ----------
    par: dictionary with keys om_b, om_m, n_s, h, w_0, sigma_8
        cosmological parameters Omega_b*h^2, Omega_m*h^2, n_s, h, w_0, sigma_8
    redshift: float
        redshift
    power_k: numpy array
        array of k values where the power spectrum will be interpolated (optional)
    
    Returns
    -------
    2 numpy arrays
        k and P(k)
    """
    
    import e2py as emu
    result = emu.get_pnonlin(par, redshift)
    power_kemu = result['k']
    power_pemu = result['P_nonlin']
    if power_k.size>0:
        power_p = np.interp(power_k,power_kemu,power_pemu)
        return power_k, power_p
    else:
        return power_kemu, power_pemu
