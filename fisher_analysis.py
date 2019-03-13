#!/usr/bin/env python
# fisher_analysis.py - Linda Blot (linda.blot@obspm.fr) - 2015
# ---------------------------------- IMPORT ---------------------------------- #
import math
import glob
import os
import sys
import numpy as np
import scipy
from read_files import *
from power_types import *
from power_stats import *
from power_covariance import *
# ---------------------------------------------------------------------------- #



# -------------------------------- FISHER -------------------------------- #
def fisher_matrix(powertype = "power", galaxy = False, list_par = [0,2,3,4],  fiducial = [0.2573,0.04356,0.963,-1.,0.801,0.], mainpath = "", simset = "", isimmin = 1, isimmax = 2, list_noutput = [1], frac=1., okprint = False, store = False):
    """ Fisher matrix. Cosmological parameters are:
    0: Omega_m * h^2
    1: Omega_b h^2
    2: n_s
    3: w
    4: sigma_8
    5: m_nu
    6-6+nz: bias for each of nz redshifts
    
    Parameters
    ----------
    powertype: string
        type of power spectrum: power, nyquist, renormalized, corrected, mcorrected, linear or linear_mock (default power)
    galaxy: bool
        matter or galaxy power spectrum
    list_par: list of int
        list of cosmological parameters to vary (default Omega_m, n_s, w, sigma_8)
    fiducial: list of float
        fiducial parameter values (default WMAP7)
    mainpath: string
        path to base folder (default empty)
    simset: string or Simset instance
        simulation set
    isimmin: int
        minimum simulation number
    isimmax: int
        maximum simulation number
    list_noutput: list of int
        list of snapshot numbers to use
    frac: float
        fraction of total galaxies observed by Euclid
    okprint: bool
        verbose (default False)
    store: bool
        store file. If True and file exists it will be overwritten (default False)
    
    Returns
    -------
    numpy array
        Fisher matrix
    """

    if (type(simset) is str):
        simset=DeusPurSet(simset)
    fltformat="%-.12e"

    nsim=isimmax-isimmin
    dalpha = 0.05
    dbeta = 0.05
    npar=len(list_par)
    fisher = np.zeros((npar,npar))
        
    for iz in range(0,len(list_noutput)):
        ioutput=list_noutput[iz]
        aexp=simset.snap_to_a(ioutput)
        redshift=1./aexp-1.
        
        # ------------------- covariance ------------------- #
        covfile = output_file_name("cov",powertype,simset,isimmin,isimmax,noutput,nmodel)
        if (os.path.isfile(covfile)):
            power_pcov=np.loadtxt(covfile,unpack=True)
            power_k,dummy=power_spectrum(powertype,mainpath,simset.name,isimmin,ioutput,aexp)
        else:
            power_k,dummy,dummy,power_pcov=cov_power(powertype,mainpath,simset.name,isimmin,isimmax,ioutput,aexp,okprint=okprint)

        if galaxy:
            simset256 = DeusPurSet("all_256")
            power_k,power_pmean,dummy=mean_power("mcorrected",mainpath,simset256.name,1,simset256.nsimmax+1,ioutput,aexp,okprint=okprint)
            bias=1.#galaxy_bias(power_k,redshift)
            ng=galaxy_density(ioutput,frac)
            
            biased_cov = pow(bias,4.)*power_pcov + 2.*pow(bias,2.)*np.sqrt(np.outer(power_pmean,power_pmean))/ng + 1./pow(ng,2.)
            inv_cov = np.linalg.inv(biased_cov)
        else:
            inv_cov = np.linalg.inv(power_pcov)
        
        if (nsim>power_k.size+2):
            inv_cov = float(nsim-power_k.size-2)*inv_cov/float(nsim-1)
        else:
            print "warning: biased inverse covariance estimator"

        # ------- variations of power spectrum wrt parameters ----- #
        derpar_T=np.zeros((npar,power_k.size))
        for ia in range(0,npar):
            ialpha=list_par[ia]
            if ((ialpha>5) and (ialpha!=6+iz)):
                derpar_T[ia]=np.zeros(power_k.size)
            else:
                if (ialpha==6+iz):
                    ialpha=6
                dummy,Ppda=pkann_power(par_list(ialpha,dalpha,1),redshift,power_k)
                dummy,Pmda=pkann_power(par_list(ialpha,dalpha,-1),redshift,power_k)
                dummy,Pp2da=pkann_power(par_list(ialpha,dalpha,2),redshift,power_k)
                dummy,Pm2da=pkann_power(par_list(ialpha,dalpha,-2),redshift,power_k)
                dtheta_alpha=dalpha*abs(fiducial[ialpha])
                derpar_T[ia]=2.*(Ppda-Pmda)/(3.*dtheta_alpha)+(Pp2da-Pm2da)/(12.*dtheta_alpha)

        fisher_iz=np.dot(derpar_T,np.dot(inv_cov,derpar_T.T))
        fisher+=fisher_iz

    if (store):
        fname = "fisher_"+powertype+"_"
        if not galaxy:
            fname += "matter.txt"
        else:
            fname += "galaxy.txt"
        f = open(fname, "w")
        for i in xrange(0, npar):
            for j in xrange(0, npar):
                f.write(str(fltformat%fisher[i,j])+" ")
            f.write("\n")
        f.close()


    return fisher



# -------------------------------- FISHER -------------------------------- #
def fisher_matrix_cho(powertype = "power", galaxy = False, list_par = [0,2,3,4],  fiducial = [0.2573,0.04356,0.963,-1.,0.801,0.], mainpath = "", simset = "", isimmin = 1, isimmax = 2, list_noutput = [1], frac=1., okprint = False, store = False):
    """ Fisher matrix computed using Cholewsky decomposition. Cosmological parameters are:
    0: Omega_m * h^2
    1: Omega_b h^2
    2: n_s
    3: w
    4: sigma_8
    5: m_nu
    6-6+nz: bias for each of nz redshifts
    
    Parameters
    ----------
    powertype: string
        type of power spectrum: power, nyquist, renormalized, corrected, mcorrected, linear or linear_mock (default power)
    galaxy: bool
        matter or galaxy power spectrum
    list_par: list of int
        list of cosmological parameters to vary (default Omega_m, n_s, w, sigma_8)
    fiducial: list of float
        fiducial parameter values (default WMAP7)
    mainpath: string
        path to base folder (default empty)
    simset: string or Simset instance
        simulation set
    isimmin: int
        minimum simulation number
    isimmax: int
        maximum simulation number
    list_noutput: list of int
        list of snapshot numbers to use
    frac: float
        fraction of total galaxies observed by Euclid
    okprint: bool
        verbose (default False)
    store: bool
        store file. If True and file exists it will be overwritten (default False)
    
    Returns
    -------
    numpy array
        Fisher matrix
    """
    
    if (type(simset) is str):
        simset=DeusPurSet(simset)
    fltformat="%-.12e"

    nsim=isimmax-isimmin
    dalpha = 0.05
    dbeta = 0.05
    npar=len(list_par)
    fisher = np.zeros((npar,npar))
    
    for iz in range(0,len(list_noutput)):
        ioutput=list_noutput[iz]
        aexp=simset.snap_to_a(ioutput)
        redshift=1./aexp-1.
        
        # ------------------- covariance ------------------- #
        covfile = output_file_name("cov",powertype,simset,isimmin,isimmax,noutput,nmodel)
        if (os.path.isfile(covfile)):
            power_pcov=np.loadtxt(covfile,unpack=True)
            power_k,dummy=power_spectrum(powertype,mainpath,simset.name,isimmin,ioutput,aexp)
        else:
            power_k,dummy,dummy,power_pcov=cov_power(powertype,mainpath,simset.name,isimmin,isimmax,ioutput,aexp,okprint=okprint)

        if galaxy:
            simset256 = DeusPurSet("all_256")
            power_k,power_pmean,dummy=mean_power("mcorrected",mainpath,simset256.name,1,simset256.nsimmax+1,ioutput,aexp,okprint=okprint)
            bias=1.#galaxy_bias(power_k,redshift)
            ng=galaxy_density(ioutput,frac)
            biased_cov = pow(bias,4.)*power_pcov + 2.*pow(bias,2.)*np.sqrt(np.outer(power_pmean,power_pmean))/ng + 1./pow(ng,2.)
            cov_fac = scipy.linalg.cho_factor(biased_cov)
        else:
            cov_fac = scipy.linalg.cho_factor(power_pcov)


        # ------- variations of power spectrum wrt parameters ----- #
        derpar_T=np.zeros((npar,power_k.size))
        for ia in range(0,npar):
            ialpha=list_par[ia]
            if ((ialpha>5) and (ialpha!=6+iz)):
                derpar_T[ia]=np.zeros(power_k.size)
            else:
                if (ialpha==6+iz):
                    ialpha=6
                dummy,Ppda=pkann_power(par_list(ialpha,dalpha,1),redshift,power_k)
                dummy,Pmda=pkann_power(par_list(ialpha,dalpha,-1),redshift,power_k)
                dummy,Pp2da=pkann_power(par_list(ialpha,dalpha,2),redshift,power_k)
                dummy,Pm2da=pkann_power(par_list(ialpha,dalpha,-2),redshift,power_k)
                dtheta_alpha=dalpha*abs(fiducial[ialpha])
                derpar_T[ia]=2.*(Ppda-Pmda)/(3.*dtheta_alpha)+(Pp2da-Pm2da)/(12.*dtheta_alpha)

        inv_cov_der =  scipy.linalg.cho_solve(cov_fac,derpar_T.T)
        if (nsim>power_k.size+2):
            inv_cov_der = float(nsim-power_k.size-2)*inv_cov_der/float(nsim-1)
        else:
            print "warning: biased inverse covariance estimator"
        fisher_iz = np.dot(derpar_T, inv_cov_der)
        fisher+=fisher_iz

    if (store):
        fname = "fisher_"+powertype+"_"
        if not galaxy:
            fname += "matter.txt"
        else:
            fname += "galaxy.txt"
        f = open(fname, "w")
        for i in xrange(0, npar):
            for j in xrange(0, npar):
                f.write(str(fltformat%fisher[i,j])+" ")
            f.write("\n")
        f.close()

    return fisher



# -------------------------------- FISHER WITH k-CUT -------------------------------- #
def fisher_matrix_kcut(kmin, kmax, powertype = "power", galaxy = False, list_par = [0,2,3,4],  fiducial = [0.2573,0.04356,0.963,-1.,0.801,0.], mainpath = "", simset = "", isimmin = 1, isimmax = 2, list_noutput = [1], frac=1., okprint = False, store = False):
    """ Fisher matrix computed using Cholewsky decomposition with k cut between kmin and kmax. Cosmological parameters are:
    0: Omega_m * h^2
    1: Omega_b h^2
    2: n_s
    3: w
    4: sigma_8
    5: m_nu
    6-6+nz: bias for each of nz redshifts
    
    Parameters
    ----------
    kmin: float
        minimum k
    kmax: float
        maximum k
    powertype: string
        type of power spectrum: power, nyquist, renormalized, corrected, mcorrected, linear or linear_mock (default power)
    galaxy: bool
        matter or galaxy power spectrum
    list_par: list of int
        list of cosmological parameters to vary (default Omega_m, n_s, w, sigma_8)
    fiducial: list of float
        fiducial parameter values (default WMAP7)
    mainpath: string
        path to base folder (default empty)
    simset: string or Simset instance
        simulation set
    isimmin: int
        minimum simulation number
    isimmax: int
        maximum simulation number
    list_noutput: list of int
        list of snapshot numbers to use
    frac: float
        fraction of total galaxies observed by Euclid
    okprint: bool
        verbose (default False)
    store: bool
        store file. If True and file exists it will be overwritten (default False)
    
    Returns
    -------
    numpy array
        Fisher matrix
    """
    
    if (type(simset) is str):
        simset=DeusPurSet(simset)
    fltformat="%-.12e"
    
    nsim=isimmax-isimmin
    dalpha = 0.05
    dbeta = 0.05
    npar=len(list_par)
    fisher = np.zeros((npar,npar))
    
    for iz in range(0,len(list_noutput)):
        ioutput=list_noutput[iz]
        aexp=simset.snap_to_a(ioutput)
        redshift=1./aexp-1.
        
        # ------------------- covariance ------------------- #
        covfile = output_file_name("cov",powertype,simset,isimmin,isimmax,noutput,nmodel)
        if (os.path.isfile(covfile)):
            power_pcov_nocut=np.loadtxt(covfile,unpack=True)
            power_k_nocut,dummy=power_spectrum(powertype,mainpath,simset.name,1,ioutput,aexp,okprint=okprint)
        else:
            power_k_nocut,dummy,dummy,power_pcov_nocut=cov_power(powertype,mainpath,simset.name,isimmin,isimmax,ioutput,aexp,okprint=okprint)
        imin = np.searchsorted(power_k_nocut,kmin)
        imax = np.searchsorted(power_k_nocut,kmax)
        power_k = power_k_nocut[imin:imax]
        power_pcov=power_pcov_nocut[imin:imax,imin:imax]
        
        if galaxy:
            simset256 = DeusPurSet("all_256")
            dummy,power_pmean_nocut,dummy=mean_power("mcorrected",mainpath,simset256.name,1,simset256.nsimmax+1,ioutput,aexp,okprint=okprint)
            power_pmean=power_pmean_nocut[imin:imax]
            biased_cov=np.zeros(np.shape(power_pcov))
            bias=1.#galaxy_bias(power_k,redshift)
            ng=galaxy_density(ioutput,frac)
            biased_cov = pow(bias,4.)*power_pcov + 2.*pow(bias,2.)*np.sqrt(np.outer(power_pmean,power_pmean))/ng + 1./pow(ng,2.)
            cov_fac = scipy.linalg.cho_factor(biased_cov)
        else:
            cov_fac = scipy.linalg.cho_factor(power_pcov)

        # ------- variations of power spectrum wrt parameters ----- #
        derpar_T=np.zeros((npar,power_k.size))
        for ia in range(0,npar):
            ialpha=list_par[ia]
            if ((ialpha>5) and (ialpha!=6+iz)):
                derpar_T[ia]=np.zeros(power_k.size)
            else:
                if (ialpha==6+iz):
                    ialpha=6
                dummy,Ppda=pkann_power(par_list(ialpha,dalpha,1),redshift,power_k)
                dummy,Pmda=pkann_power(par_list(ialpha,dalpha,-1),redshift,power_k)
                dummy,Pp2da=pkann_power(par_list(ialpha,dalpha,2),redshift,power_k)
                dummy,Pm2da=pkann_power(par_list(ialpha,dalpha,-2),redshift,power_k)
                dtheta_alpha=dalpha*abs(fiducial[ialpha])
                derpar_T_nocut=2.*(Ppda-Pmda)/(3.*dtheta_alpha)+(Pp2da-Pm2da)/(12.*dtheta_alpha)
                derpar_T[ia]=derpar_T_nocut[imin:imax]
        
        inv_cov_der =  scipy.linalg.cho_solve(cov_fac,derpar_T.T)
        if (nsim>power_k.size+2):
            inv_cov_der = float(nsim-power_k.size-2)*inv_cov_der/float(nsim-1)
        else:
            print "warning: biased inverse covariance estimator"
        fisher_iz = np.dot(derpar_T, inv_cov_der)
        fisher+=fisher_iz
            
    if (store):
        fname = "fisher_"+powertype+"_k"+str(kmin)+"_"+str(kmax)
        if not galaxy:
            fname += "matter.txt"
        else:
            fname += "galaxy.txt"
        f = open(fname, "w")
        for i in xrange(0, npar):
            for j in xrange(0, npar):
                f.write(str(fltformat%fisher[i,j])+" ")
            f.write("\n")
        f.close()


    return fisher

def par_list(ipar,dpar,fact):
    """ Get cosmological parameter list for PkANN power spectra
        
    Parameters
    ----------
    ipar: int
        parameter number
    dpar: float
        amount of variation around fiducial
    fact: int
        number of times the variation is applied (can be negative)
    
    Returns
    -------
    list
        parameter list
    """
    
    # omega_m h^2, omega_b h^2, n_s, w, sigma_8, m_nu, bias
    par=np.array([0.2573*0.5184, 0.04356*0.5184, 0.963, -1., 0.801, 0., 1.])
    par[ipar] += fact * dpar * abs(par[ipar])
    return par

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
