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
def fisher_matrix(powertype = "power", galaxy = 0, list_par = [0,2,3,4],  fiducial = [0.2573,0.04356,0.963,-1.,0.801,0.], mainpath = "", simset = "", isimmin = 1, isimmax = 2, list_noutput = [1], list_aexp = [0.], growth_a = np.zeros(0), growth_dplus = np.zeros(0), frac=1., okprint = False):
    
    nsim=isimmax-isimmin
    dalpha = 0.05
    dbeta = 0.05
    fisher = np.zeros((len(list_par),len(list_par)))
    npar=len(list_par)

    for iz in range(0,len(list_aexp)):
        ioutput=list_noutput[iz]
        aexp=list_aexp[iz]
        
        # ------------------- covariance ------------------- #
        covfile = "tmp/cov_"+str(nsim)+"_"+powertype+"_"+str(ioutput)+".txt"
        if (os.path.isfile(covfile)):
            power_pcov=np.loadtxt(covfile,unpack=True)
            power_k,dummy=power_spectrum(powertype,mainpath,simset,isimmin,ioutput,aexp,growth_a,growth_dplus)
        else:
            power_k,dummy,dummy,power_pcov=cov_power(powertype,mainpath,simset,isimmin,isimmax,ioutput,aexp,growth_a,growth_dplus)
            
            if (galaxy < 2):
                fltformat="%-.12e"
                fout = open(covfile,"w")
                for ik in range(0, power_k.size):
                    for jk in range(0, power_k.size):
                        fout.write(str(fltformat%power_pcov[ik,jk])+" ")
                    fout.write("\n")
                fout.close()

        if (galaxy > 0):
            power_k,power_pmean,dummy=mean_power_all("mcorrected",mainpath,ioutput,aexp,growth_a,growth_dplus)
            biased_cov=np.zeros(np.shape(power_pcov))
            bias=1.#galaxy_bias(power_k,ioutput)
            ng=galaxy_density(ioutput,frac)
            for ik in range(0,power_k.size):
                for jk in range(0,power_k.size):
                    biased_cov[ik,jk]=pow(bias,4.)*power_pcov[ik,jk]+2.*pow(bias,2.)*math.sqrt(power_pmean[ik]*power_pmean[jk])/ng+1./(ng*ng)
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

    return fisher



# -------------------------------- FISHER -------------------------------- #
def fisher_matrix_cho(powertype = "power", galaxy = 0, list_par = [0,2,3,4],  fiducial = [0.2573,0.04356,0.963,-1.,0.801,0.], mainpath = "", simset = "", isimmin = 1, isimmax = 2, list_noutput = [1], list_aexp = [0.], growth_a = np.zeros(0), growth_dplus = np.zeros(0), frac=1., okprint = False):
    
    nsim=isimmax-isimmin
    dalpha = 0.05
    dbeta = 0.05
    fisher = np.zeros((len(list_par),len(list_par)))
    npar=len(list_par)
    
    for iz in range(0,len(list_aexp)):
        ioutput=list_noutput[iz]
        aexp=list_aexp[iz]
        
        # ------------------- covariance ------------------- #
        covfile = "tmp/cov_"+str(nsim)+"_"+powertype+"_"+str(ioutput)+".txt"
        if (os.path.isfile(covfile)):
            power_pcov=np.loadtxt(covfile,unpack=True)
        else:
            power_k,dummy,dummy,power_pcov=cov_power(powertype,mainpath,simset,isimmin,isimmax,ioutput,aexp,growth_a,growth_dplus)
            
            if (galaxy < 2):
                fltformat="%-.12e"
                fout = open(covfile,"w")
                for ik in range(0, power_k.size):
                    for jk in range(0, power_k.size):
                        fout.write(str(fltformat%power_pcov[ik,jk])+" ")
                    fout.write("\n")
                fout.close()

        if (galaxy > 0):
            power_k,power_pmean,dummy=mean_power_all("mcorrected",mainpath,ioutput,aexp,growth_a,growth_dplus)
            biased_cov=np.zeros(np.shape(power_pcov))
            bias=1.#galaxy_bias(power_k,ioutput)
            ng=galaxy_density(ioutput,frac)
            for ik in range(0,power_k.size):
                for jk in range(0,power_k.size):
                    biased_cov[ik,jk]=pow(bias,4.)*power_pcov[ik,jk]+2.*pow(bias,2.)*math.sqrt(power_pmean[ik]*power_pmean[jk])/ng+1./(ng*ng)
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

    return fisher



# -------------------------------- FISHER WITH k-CUT -------------------------------- #
def fisher_matrix_kcut(kmin, kmax, powertype = "power", galaxy = 0, list_par = [0,2,3,4],  fiducial = [0.2573,0.04356,0.963,-1.,0.801,0.], mainpath = "", simset = "", isimmin = 1, isimmax = 2, list_noutput = [1], list_aexp = [0.], growth_a = np.zeros(0), growth_dplus = np.zeros(0), frac=1., okprint = False):
    
    nsim=isimmax-isimmin
    dalpha = 0.05
    dbeta = 0.05
    fisher = np.zeros((len(list_par),len(list_par)))
    npar=len(list_par)
    
    for iz in range(0,len(list_aexp)):
        ioutput=list_noutput[iz]
        aexp=list_aexp[iz]
        
        power_k_nocut,dummy=power_spectrum(powertype,mainpath,simset,1,ioutput,aexp,growth_a,growth_dplus,False)
        idx=[(power_k_nocut > kmin) & (power_k_nocut < kmax)]
        power_k=power_k_nocut[idx]
        imin=np.where(power_k_nocut==power_k[0])[0]
        imax=np.where(power_k_nocut==power_k[power_k.size-1])[0]+1
        
        # ------------------- covariance ------------------- #
        covfile = "tmp/cov_"+str(nsim)+"_"+powertype+"_"+str(ioutput)+".txt"
        if (os.path.isfile(covfile)):
            power_pcov_nocut=np.loadtxt(covfile,unpack=True)
        else:
            dummy,dummy,dummy,power_pcov_nocut=cov_power(powertype,mainpath,simset,isimmin,isimmax,ioutput,aexp,growth_a,growth_dplus)
            
            if (galaxy < 2):
                fltformat="%-.12e"
                fout = open(covfile,"w")
                for ik in range(0, power_k.size):
                    for jk in range(0, power_k.size):
                        fout.write(str(fltformat%power_pcov_nocut[ik,jk])+" ")
                    fout.write("\n")
                fout.close()

        power_pcov=power_pcov_nocut[imin:imax,imin:imax]
        if (galaxy > 0):
            dummy,power_pmean_nocut,dummy=mean_power_all("mcorrected",mainpath,ioutput,aexp,growth_a,growth_dplus)
            power_pmean=power_pmean_nocut[idx]
            biased_cov=np.zeros(np.shape(power_pcov))
            bias=1.#galaxy_bias(power_k,ioutput)
            ng=galaxy_density(ioutput,frac)
            for ik in range(0,power_k.size):
                for jk in range(0,power_k.size):
                    biased_cov[ik,jk]=pow(bias,4.)*power_pcov[ik,jk]+2.*pow(bias,2.)*math.sqrt(power_pmean[ik]*power_pmean[jk])/ng+1./(ng*ng)
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

    return fisher

def galaxy_bias(power_k=np.zeros(0), noutput=1):
    bias=np.zeros(power_k.size)
    for ik in xrange(0,power_k.size):
        bias[ik]=1.5
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
