#!/usr/bin/env python2.7
# fisher_analysis.py - Linda Blot (linda.blot@obspm.fr) - 2015
# ---------------------------------- IMPORT ---------------------------------- #
import os
import numpy as np
import scipy
from read_files import *
from power_types import *
from power_stats import *
from power_covariance import *
# ---------------------------------------------------------------------------- #



# -------------------------------- FISHER -------------------------------- #
def fisher_matrix(simset = DeusPurSet("all_256"), list_par = ['om_m','n_s','w_0','sigma_8'], list_noutput = None,  fiducial = None, galaxy = False, derivative = "pkann", cholewsky = False, hartlap=False, powertype = "power", mainpath = "", isimmin = 1, isimmax = None, frac=1., kmin = None, kmax = None, dtheta = 0.05, okprint = False, store = False, outpath = "./", test_diag = False, test_gaussian = False, test_diag_gal = False):
    """ Function to compute the Fisher matrix using the covariance estimated from a set of simulations. The derivatives can be computed using an emulator or passed as argument

    Parameters
    ----------
    simset: Simset instance
        simulation set used to compute the covariance (default DeusPurSet("all_256"))
    list_par: list of simset.cosmo_par keys
        list of cosmological parameters to vary (default Omega_m, n_s, w, sigma_8)
    list_noutput: list of int
        list of snapshot numbers to use (default all snapshots of simset)
    fiducial: simset.cosmo_par dictionary
        fiducial parameter values (default simset.cosmo_par)
    galaxy: bool
        matter or galaxy power spectrum (default True)
    derivative: string
        emulator name ("pkann" or "euemu") or "sim" for numerical derivative from simulation sets
    cholewski: bool
        use Cholewski decomposition to compute Fisher matrix (default False)
    hartlap: bool
        use Hartlap correction for the inverse of the covariance (default False)
    powertype: string
        type of power spectrum: power, nyquist, renormalized, corrected, mcorrected, linear or linear_mock (default power)
    mainpath: string
        path to base folder (default empty)
    isimmin: int
        minimum simulation number (default 1)
    isimmax: int
        maximum simulation number (default simset.nsimmax)
    frac: float
        fraction of total galaxies observed by Euclid (default 1)
    kmin: float
        minimum k (default None)
    kmax: float
        maximum k (default None)
    dtheta: float
        fraction of parameter value for derivative variation (default 0.05, ignored if simulation derivative)
    okprint: bool
        verbose (default False)
    store: bool
        store file. If True and file exists it will be overwritten (default False)
    outpath: string
        path where output file is stored (default .)
    test_diag: bool
        use correlation coefficient of the fiducial (default False)
    test_gaussian: bool
        use Gaussian covariance (default False)

    Returns
    -------
    Pandas DataFrame
        Fisher matrix
    """

    # Set default values
    if fiducial==None:
        fiducial = simset.cosmo_par
        if galaxy:
            fiducial.update({'bias':1.})
    if list_noutput==None:
        list_noutput = range(1,len(simset.alist))
        print list_noutput
    if isimmax==None:
        isimmax = simset.nsimmax

    fltformat="%-.12e"
    nsim=isimmax-isimmin
    dalpha = dtheta
    dbeta = dtheta
    npar=len(list_par)
    fisher = np.zeros((npar,npar))

    # ------------------- Loop over redshift  ------------------- #
    for ioutput in list_noutput:
        aexp=simset.snap_to_a(ioutput)
        redshift=1./aexp-1.
        # ------------------- Data covariance ------------------- #
        if test_gaussian:
            power_k, power_pmean, dummy = mean_power(powertype, mainpath, simset, isimmin, isimmax, ioutput, aexp, okprint=okprint, outpath=outpath)
            nk = simset.num_modes(power_k)
            power_pcov = np.diag(2./nk*np.power(power_pmean,2))
        else:
            power_k,dummy,dummy,power_pcov=cov_power(powertype, mainpath, simset, isimmin, isimmax, ioutput, aexp, okprint=okprint, outpath=outpath)
        ksize = power_k.size
        if test_diag:
            simset_fid = DeusPurSet("512_adaphase_512_328-125", 2, datapath=simset.datapath)
            dummy,dummy,power_psigma_fid,power_pcov_fid=cov_power(powertype, mainpath, simset_fid, isimmin, isimmax, ioutput, aexp, okprint=okprint, outpath=outpath)
            power_psigma = np.sqrt(np.diag(power_pcov))
            power_pcov=power_pcov/np.outer(power_psigma,power_psigma)*np.outer(power_psigma_fid,power_psigma_fid)
        # Cut scales if necessary
        if kmin!=None or kmax!=None:
            if kmin==None:
                imin = 0
            else:
                imin = np.searchsorted(power_k,kmin)
            if kmax==None:
                imax = -1
            else:
                imax = np.searchsorted(power_k,kmax)
            power_k = power_k[imin:imax]
            power_pcov=power_pcov[imin:imax,imin:imax]
        # ------------------- Biased tracers ------------------- #
        if galaxy:
            simset256 = DeusPurSet("all_256")
            power_k, power_pmean, dummy = mean_power("mcorrected", mainpath, simset256, 1, simset256.nsimmax, ioutput, aexp, okprint=okprint, outpath=outpath)
            # Cut scales if necessary
            if kmin!=None or kmax!=None:
                power_k = power_k[imin:imax]
                power_pmean = power_pmean[imin:imax]
            bias=1.#galaxy_bias(power_k,redshift)
            ng=galaxy_density(ioutput,frac)
            biased_cov = pow(bias,4.)*power_pcov #+ (2.*pow(bias,2.)*power_pmean/ng + 1./pow(ng,2.))*np.eye(power_pcov.shape[0]) #np.sqrt(np.outer(power_pmean,power_pmean))
            if test_diag_gal:
                biased_cov = np.diag(np.diag(biased_cov))
            if cholewsky:
                cov_fac = scipy.linalg.cho_factor(biased_cov)
            else:
                inv_cov = np.linalg.inv(biased_cov)
        else:
            if cholewsky:
                cov_fac = scipy.linalg.cho_factor(power_pcov)
            else:
                inv_cov = np.linalg.inv(power_pcov)

        # ------------------- Derivatives ------------------- #
        if derivative=="pkann":
            derpar = pkann_derivative(fiducial, list_par, dtheta, redshift, power_k)
        elif derivative=="euemu":
            raise NotImplementedError("Euclid emulator derivatives not implemented")
        elif derivative=="sim":
            derpar = sim_derivative(powertype, mainpath, ioutput, list_par, ksize, outpath = outpath, datapath = simset.datapath)
            # Cut scales if necessary
            if kmin!=None or kmax!=None:
                if kmin==None:
                    imin = 0
                else:
                    imin = np.searchsorted(power_k,kmin)
                if kmax==None:
                    imax = -1
                else:
                    imax = np.searchsorted(power_k,kmax)
                derpar = derpar[imin:imax,:]
        else:
            raise ValueError("Unknown emulator type "+derivative)
       # ------------------- Fisher matrix computation ------------------- #
        if cholewsky:
            inv_cov_der =  scipy.linalg.cho_solve(cov_fac,derpar)
        else:
            inv_cov_der = np.dot(inv_cov,derpar)
        if hartlap and nsim>power_k.size+2:
            inv_cov_der = float(nsim-power_k.size-2)*inv_cov_der/float(nsim-1)
        fisher_iz = np.dot(derpar.T, inv_cov_der)
        fisher+=fisher_iz

    if (store):
        fname = outpath+"/fisher_"+powertype+"_"
        if not galaxy:
            fname += "matter.txt"
        else:
            fname += "galaxy.txt"
        if okprint:
            print "Writing file ", fname
        f = open(fname, "w")
        for i in xrange(0, npar):
            for j in xrange(0, npar):
                f.write(str(fltformat%fisher[i,j])+" ")
            f.write("\n")
        f.close()

    return pd.DataFrame(fisher, index=list_par, columns=list_par)

def sim_derivative(powertype, mainpath, noutput, list_par, ksize, outpath="./", datapath="/data/deus_pur_cosmo/data/"):
    npar = len(list_par)
    derpar_T = np.zeros((npar,ksize))
    for ia,param in enumerate(list_par):
        simsetf = DeusPurSet("512_adaphase_512_328-125",cosmo_iterator(param,'fid'),datapath=datapath)
        if param=='h':
            kf, dummy, dummy = mean_power(powertype, mainpath, simsetf, 1, simsetf.nsimmax+1, noutput, aexp, outpath=outpath)
        pks = {}
        for variation in ['m','p']:
            simset = DeusPurSet("512_adaphase_512_328-125",cosmo_iterator(param,variation),datapath=datapath)
            aexp = simset.snap_to_a(noutput)
            k, pk, dummy = mean_power(powertype, mainpath, simset, 1, simset.nsimmax+1, noutput, aexp, outpath=outpath)
            if param=='h':
                pk = pk * simset.cosmo_par['h']**3 / simsetf.cosmo_par['h']**3
                pk = np.interp(kf, k, pk)
            pks.update({variation:pk})
        dtheta_par = np.abs(simset.cosmo_par[param]-simsetf.cosmo_par[param])
        derpar_T[ia] = 0.5*(pks['p']-pks['m'])/dtheta_par
    return derpar_T.T

def pkann_derivative(fiducial, list_par, dtheta, redshift, power_k):
    npar=len(list_par)
    derpar_T = np.zeros((npar,power_k.size))
    for ia,param in enumerate(list_par):
        dummy,Ppda = pkann_power(pkann_par_list(fiducial,param,dtheta,1),redshift,power_k)
        dummy,Pmda = pkann_power(pkann_par_list(fiducial,param,dtheta,-1),redshift,power_k)
        dummy,Pp2da = pkann_power(pkann_par_list(fiducial,param,dtheta,2),redshift,power_k)
        dummy,Pm2da = pkann_power(pkann_par_list(fiducial,param,dtheta,-2),redshift,power_k)
        dtheta_par = dtheta*abs(fiducial[param])
        derpar_T[ia] = 2.*(Ppda-Pmda)/(3.*dtheta_par)+(Pp2da-Pm2da)/(12.*dtheta_par)
    return derpar_T.T

def pkann_par_list(fiducial,param,dpar,fact):
    """ Get cosmological parameter list for PkANN power spectra

    Parameters
    ----------
    fiducial: simset.cosmo_par dictionary
        fiducial values of the parameters
    param: string
        simset.cosmo_par dictionary key
    dpar: float
        amount of variation around fiducial
    fact: int
        number of times the variation is applied (can be negative)

    Returns
    -------
    list
        parameter list
    """
    cosmo = fiducial.copy()
    if 'bias' not in fiducial.keys():
        cosmo.update({'bias':1.})
    cosmo.update({param: fiducial[param] + fact * dpar * abs(fiducial[param])})
    return cosmo

def galaxy_bias(power_k, redshift=1.):
    """ Fiducial galaxy bias for given k bins and redshift

    Parameters
    ----------
    power_k: numpy array of floats
        k bins
    redshift: float
        redshift (default 1.)

    Returns
    -------
    numpy array of floats
        fiducial galaxy bias
    """

    bias=np.zeros(power_k.size)
    bias=np.fill(1.5)
    return bias


def galaxy_density(noutput=1,frac=1.):
    """ Galaxy number density observed by Euclid

    Parameters
    ----------
    noutput: int
        snapshot number
    frac: float
        fraction of observed galaxies

    Returns
    -------
    float
        galaxy number density in (h/Mpc)^3
    """

    if (noutput==2):
        ng=frac*1.5e-4
    elif (noutput==3):
        ng=frac*7.7e-4
    elif (noutput==4):
        ng=frac*1.81e-3
    elif (noutput==5):
        ng=frac*2.99e-3
    elif (noutput==6):
        ng=frac*4.2e-3
    else:
        print "redshift bin outside Euclid capabilities"
        ng=5.e-4
    return ng
