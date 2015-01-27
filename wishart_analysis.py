#!/usr/bin/env python
# wishart_analysis.py - Linda Blot (linda.blot@obspm.fr) - 2014
# ---------------------------------- IMPORT ---------------------------------- #
import math
import glob
import os
import sys
import numpy as np
import scipy.interpolate as itl
from scipy import stats
from numpy import linalg
from read_files import *
from power_types import *
from power_stats import *
from power_covariance import *
from math import sqrt as sqrt
#import os.path
# ---------------------------------------------------------------------------- #



# --------------------------- SAMPLE COVARIANCE VARIANCE --------------------------- #
def cov_variance(powertype = "power", mainpath = "", noutput = 1 , aexp = 1., growth_a = np.zeros(0), growth_dplus = np.zeros(0), list_nr = [10,50,100,500,700,1000,3000,5000,6000], okprint=False):

    simset = "all_256"
    trace_sigma2=np.zeros(len(list_nr))
    i=0 #index on list_nr
    
    power_k,dummy = power_spectrum(powertype,mainpath,"4096_adaphase_256",1,noutput,aexp,growth_a,growth_dplus)
    nbin = power_k.size

    for nr in list_nr:

        nsub = 12288/nr
        power_pmean = np.zeros((nsub,nbin))
        power_pvar = np.zeros((nsub,nbin))
        nsim = 0
        totsim = 12288 - int(math.fmod(12288,nr))
        
        print nr,totsim

        for isub in range(0, nsub):
            isimmin = isub * nr + 1
            isimmax = (isub+1)*nr
            
            dummy,dummy,power_psigma=mean_power(powertype, mainpath, simset, isimmin, isimmax+1, noutput, aexp, growth_a, growth_dplus)
                
            for ik in range(0,nbin):
                power_pvar[isub][ik]=power_psigma[ik]*power_psigma[ik]

        var_mean = np.zeros(nbin)
        for ik in range(0,nbin):
            for isubset in range(0,nsub):
                var_mean[ik] += power_pvar[isubset][ik]
        var_mean /= float(nsub)

        sigma2 = np.zeros(nbin)
        fact = 0.
        for ik in range(0,nbin):
            for isubset in range(0,nsub):
                sigma2[ik]+=(power_pvar[isubset][ik] - var_mean[ik]) * (power_pvar[isubset][ik] - var_mean[ik])
            fact+=var_mean[ik]*var_mean[ik]
            trace_sigma2[i]+=sigma2[ik]
        trace_sigma2[i]/=(fact*float(nsub-1))

        i+=1
    return list_nr, trace_sigma2
# ---------------------------------------------------------------------------- #



# --------------------------- PRECISION MATRIX VARIANCE --------------------------- #
def inv_cov_variance(powertype = "power", mainpath = "", noutput = 1 , aexp = 1., growth_a = np.zeros(0), growth_dplus = np.zeros(0), list_nr = [300,500,700,1000,3000,5000,6000], okprint=False):

    simset = "all_256"
    trace_sigma2=np.zeros(len(list_nr))
    i=0

    power_k,dummy = power_spectrum(powertype,mainpath,"4096_adaphase_256",1,noutput,aexp,growth_a,growth_dplus)

    for nr in list_nr:
        nsub = 12288/nr
        nbin = power_k.size
        var_inv = np.zeros((nsub,nbin))
        totsim = 12288 - int(math.fmod(12288,nr))
        
        print nr,totsim

        var_all = np.zeros(nbin)
        var_inv_mean = np.zeros(nbin)
        for isub in range(0, nsub):
            isimmin = isub * nr + 1
            isimmax = (isub+1)*nr
            #if(isub==nsub-1):
                #print isub, isimmin, isimmax
            filename="tmp/"+str("%05d"%noutput)+"/inv_cov_"+powertype+"_"+str("%05d"%nr)+"_"+str("%05d"%isub)+".txt"
            if(os.path.isfile(filename)):
                cov_inv=np.loadtxt(filename, unpack=True)
                #cov_inv= float(nr-1)*cov_inv/float(nr-nbin-2)
            else:
                kcov,dummy,dummy,cov=cov_power(powertype, mainpath, simset, isimmin, isimmax+1, noutput, aexp, growth_a, growth_dplus)
                cov_inv=linalg.inv(cov)
                #cov_inv= float(nr-nbin-2)*cov_inv/float(nr-1)
                fltformat="%-.12e"
                f = open(filename, "w")
                for ik in range(0, nbin):
                    for jk in range(0, nbin):
                        f.write(str(fltformat%cov_inv[ik][jk])+" ")
                    f.write("\n")
                f.close()

            for ik in range(0,nbin):
                var_inv[isub][ik] = cov_inv[ik,ik]
                var_inv_mean[ik] += var_inv[isub][ik]
        var_inv_mean /= float(nsub)
        
        fact=0.
        sigma2=np.zeros(nbin)
        for ik in range(0,nbin):
            for isub in range(0,nsub):
                sigma2[ik]+=(var_inv[isub][ik] - var_inv_mean[ik]) * (var_inv[isub][ik] - var_inv_mean[ik])
                #sigma2[ik]+=(var_inv[isub][ik] - var_all[ik]) * (var_inv[isub][ik] - var_all[ik])
            #fact+= var_all[ik] * var_all[ik]
            fact += var_inv_mean[ik] * var_inv_mean[ik]
            trace_sigma2[i]+=sigma2[ik]

        trace_sigma2[i]/=(fact*float(nsub-1))

        i+=1
    return list_nr, trace_sigma2
# ---------------------------------------------------------------------------- #



# --------------------------- PRECISION MATRIX BIAS --------------------------- #
def inv_cov_bias(powertype = "power", mainpath = "", noutput = 1 , aexp = 1., growth_a = np.zeros(0), growth_dplus = np.zeros(0), list_nr = [300,500,700,1000,3000,5000,6000], okprint=False):

    simset = "all_256"
    bias=np.zeros(len(list_nr))
    i=0

    power_k, power_pmean, power_psigma, power_pcov = cov_power(powertype, mainpath, simset,1,12289,noutput,aexp,growth_a,growth_dplus)
    cov_inv_all=linalg.inv(power_pcov)
    trace_all=0.
    for ik in range(0,power_k.size):
        trace_all+=cov_inv_all[ik][ik]
    print trace_all

    for nr in list_nr:
        nsub = 12288/nr
        nbin = power_k.size
        totsim = 12288 - int(math.fmod(12288,nr))
        
        print nr,totsim

        #var_inv_mean = np.zeros(nbin)
        for isub in range(0, nsub):
            isimmin = isub * nr + 1
            isimmax = (isub+1)*nr
            filename="tmp/"+str("%05d"%noutput)+"/inv_cov_"+powertype+"_"+str("%05d"%nr)+"_"+str("%05d"%isub)+".txt"
            if(os.path.isfile(filename)):
                cov_inv=np.loadtxt(filename, unpack=True)
            else:
                kcov,dummy,dummy,cov=cov_power(powertype, mainpath, simset, isimmin, isimmax+1, noutput, aexp, growth_a, growth_dplus)
                cov_inv=linalg.inv(cov)
                fltformat="%-.12e"
                f = open(filename, "w")
                for ik in range(0, nbin):
                    for jk in range(0, nbin):
                        f.write(str(fltformat%cov_inv[ik][jk])+" ")
                    f.write("\n")
                f.close()


            #for ik in range(0,nbin):
                #var_inv_mean[ik]+=cov_inv[ik][ik]
        #var_inv_mean /= float(nsub)

        trace=0.
        for ik in range(0,nbin):
            trace+=cov_inv[ik][ik]
        print trace
        bias[i]=(trace-trace_all)/trace_all
        i+=1
    return list_nr, bias
# ---------------------------------------------------------------------------- #