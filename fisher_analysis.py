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
def fisher_matrix(powertype = "power", galaxy = 0, list_par = [0,2,3,4],  fiducial = [0.2573,0.04356,0.963,-1.,0.801,0.], mainpath = "", simset = "", isimmin = 1, isimmax = 2, list_noutput = [1], list_aexp = [0.], growth_a = np.zeros(0), growth_dplus = np.zeros(0), frac=1., okprint = False, store = False):

    if (type(simset) is str):
        simset=DeusPurSet(simset)

    nsim=isimmax-isimmin
    dalpha = 0.05
    dbeta = 0.05
    npar=len(list_par)
    fisher = np.zeros((npar,npar))

    for iz in range(0,len(list_aexp)):
        ioutput=list_noutput[iz]
        aexp=list_aexp[iz]
        
        # ------------------- covariance ------------------- #
        covfile = file_name("cov",powertype,simset,isimmin,isimmax,noutput,nmodel)
        if (os.path.isfile(covfile)):
            power_pcov=np.loadtxt(covfile,unpack=True)
            power_k,dummy=power_spectrum(powertype,mainpath,simset.name,isimmin,ioutput,aexp,growth_a,growth_dplus)
        else:
            power_k,dummy,dummy,power_pcov=cov_power(powertype,mainpath,simset.name,isimmin,isimmax,ioutput,aexp,growth_a,growth_dplus,okprint=okprint,store=store)

        if (galaxy > 0):
            simset256 = DeusPurSet("all_256")
            power_k,power_pmean,dummy=mean_power("mcorrected",mainpath,simset256.name,1,simset256.nsimmax+1,ioutput,aexp,growth_a,growth_dplus,okprint=okprint,store=store)
            bias=1.#galaxy_bias(power_k,ioutput)
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
                dummy,Ppda=pkann_power(ialpha,dalpha,1,powertype,ioutput,mainpath,aexp,growth_a,growth_dplus)
                dummy,Pmda=pkann_power(ialpha,dalpha,-1,powertype,ioutput,mainpath,aexp,growth_a,growth_dplus)
                dummy,Pp2da=pkann_power(ialpha,dalpha,2,powertype,ioutput,mainpath,aexp,growth_a,growth_dplus)
                dummy,Pm2da=pkann_power(ialpha,dalpha,-2,powertype,ioutput,mainpath,aexp,growth_a,growth_dplus)
                dtheta_alpha=dalpha*abs(fiducial[ialpha])
                derpar_T[ia]=2.*(Ppda-Pmda)/(3.*dtheta_alpha)+(Pp2da-Pm2da)/(12.*dtheta_alpha)

        fisher_iz=np.dot(derpar_T,np.dot(inv_cov,derpar_T.T))
        fisher+=fisher_iz

    if (store):
        fname = "fisher_"+powertype+"_"
        if (galaxy==0):
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
def fisher_matrix_cho(powertype = "power", galaxy = 0, list_par = [0,2,3,4],  fiducial = [0.2573,0.04356,0.963,-1.,0.801,0.], mainpath = "", simset = "", isimmin = 1, isimmax = 2, list_noutput = [1], list_aexp = [0.], growth_a = np.zeros(0), growth_dplus = np.zeros(0), frac=1., okprint = False, store = False):
    
    if (type(simset) is str):
        simset=DeusPurSet(simset)

    nsim=isimmax-isimmin
    dalpha = 0.05
    dbeta = 0.05
    npar=len(list_par)
    fisher = np.zeros((npar,npar))
    
    for iz in range(0,len(list_aexp)):
        ioutput=list_noutput[iz]
        aexp=list_aexp[iz]
        
        # ------------------- covariance ------------------- #
        covfile = file_name("cov",powertype,simset,isimmin,isimmax,noutput,nmodel)
        if (os.path.isfile(covfile)):
            power_pcov=np.loadtxt(covfile,unpack=True)
        else:
            power_k,dummy,dummy,power_pcov=cov_power(powertype,mainpath,simset.name,isimmin,isimmax,ioutput,aexp,growth_a,growth_dplus,okprint=okprint,store=store)

        if (galaxy > 0):
            simset256 = DeusPurSet("all_256")
            power_k,power_pmean,dummy=mean_power("mcorrected",mainpath,simset256.name,1,simset256.nsimmax+1,ioutput,aexp,growth_a,growth_dplus,okprint=okprint,store=store)
            bias=1.#galaxy_bias(power_k,ioutput)
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
                dummy,Ppda=pkann_power(ialpha,dalpha,1,powertype,ioutput,mainpath,aexp,growth_a,growth_dplus)
                dummy,Pmda=pkann_power(ialpha,dalpha,-1,powertype,ioutput,mainpath,aexp,growth_a,growth_dplus)
                dummy,Pp2da=pkann_power(ialpha,dalpha,2,powertype,ioutput,mainpath,aexp,growth_a,growth_dplus)
                dummy,Pm2da=pkann_power(ialpha,dalpha,-2,powertype,ioutput,mainpath,aexp,growth_a,growth_dplus)
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
        if (galaxy==0):
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
def fisher_matrix_kcut(kmin, kmax, powertype = "power", galaxy = 0, list_par = [0,2,3,4],  fiducial = [0.2573,0.04356,0.963,-1.,0.801,0.], mainpath = "", simset = "", isimmin = 1, isimmax = 2, list_noutput = [1], list_aexp = [0.], growth_a = np.zeros(0), growth_dplus = np.zeros(0), frac=1., okprint = False, store = False):
    
    if (type(simset) is str):
        simset=DeusPurSet(simset)
    
    nsim=isimmax-isimmin
    dalpha = 0.05
    dbeta = 0.05
    npar=len(list_par)
    fisher = np.zeros((npar,npar))
    
    for iz in range(0,len(list_aexp)):
        ioutput=list_noutput[iz]
        aexp=list_aexp[iz]
        
        # ------------------- covariance ------------------- #
        covfile = file_name("cov",powertype,simset,isimmin,isimmax,noutput,nmodel)
        if (os.path.isfile(covfile)):
            power_pcov_nocut=np.loadtxt(covfile,unpack=True)
            power_k_nocut,dummy=power_spectrum(powertype,mainpath,simset.name,1,ioutput,aexp,growth_a,growth_dplus,okprint=okprint)
        else:
            power_k_nocut,dummy,dummy,power_pcov_nocut=cov_power(powertype,mainpath,simset.name,isimmin,isimmax,ioutput,aexp,growth_a,growth_dplus,okprint=okprint,store=store)
        imin = np.searchsorted(power_k_nocut,kmin)
        imax = np.searchsorted(power_k_nocut,kmax)
        power_k = power_k_nocut[imin:imax]
        power_pcov=power_pcov_nocut[imin:imax,imin:imax]
        
        if (galaxy > 0):
            simset256 = DeusPurSet("all_256")
            dummy,power_pmean_nocut,dummy=mean_power("mcorrected",mainpath,simset256.name,1,simset256.nsimmax+1,ioutput,aexp,growth_a,growth_dplus,okprint=okprint,store=store)
            power_pmean=power_pmean_nocut[imin:imax]
            biased_cov=np.zeros(np.shape(power_pcov))
            bias=1.#galaxy_bias(power_k,ioutput)
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
                dummy,Ppda=pkann_power(ialpha,dalpha,1,powertype,ioutput,mainpath,aexp,growth_a,growth_dplus)
                dummy,Pmda=pkann_power(ialpha,dalpha,-1,powertype,ioutput,mainpath,aexp,growth_a,growth_dplus)
                dummy,Pp2da=pkann_power(ialpha,dalpha,2,powertype,ioutput,mainpath,aexp,growth_a,growth_dplus)
                dummy,Pm2da=pkann_power(ialpha,dalpha,-2,powertype,ioutput,mainpath,aexp,growth_a,growth_dplus)
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
        if (galaxy==0):
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

def galaxy_bias(power_k=np.zeros(0), noutput=1):
    bias=np.zeros(power_k.size)
    bias=np.fill(1.5)
    return bias


def galaxy_density(noutput=1,frac=1.):
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
