#!/usr/bin/env python
# power_covariance.py - Linda Blot (linda.blot@obspm.fr) - 2013
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
from math import sqrt as sqrt
#import os.path
# ---------------------------------------------------------------------------- #

"""@package covariance
compute power spectrum covariance matrix, precision matrix, signal-to-noise 
"""
# -------------------------------- COVARIANCE POWER -------------------------------- #
def cov_power(powertype = "power", mainpath = "", simset = "", isimmin = 1, isimmax = 2, noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), okprint = False):
    n_sim = isimmax-isimmin
    if ((simset=="all_256" and n_sim==12288) or (simset=="all_1024" and n_sim==96)):
        fname = "cov_"+simset+"_"+powertype+"_"+str("%05d"%noutput)+".txt"
        if(os.path.isfile(fname)):
            power_pcov = np.loadtxt(fname,unpack=True)
            if (simset=="all_256"):
                power_k, power_pmean, power_psigma = mean_power_all(powertype, mainpath, noutput, aexp, growth_a, growth_dplus, okprint)
            elif (simset=="all_1024"):
                power_k, power_pmean, power_psigma = mean_power_all_1024(powertype, mainpath, noutput, aexp, growth_a, growth_dplus, okprint)
        
        else:
            nsim = 0
            if (simset=="all_256"):
                power_k, power_pmean, power_psigma = mean_power_all(powertype, mainpath, noutput, aexp, growth_a, growth_dplus, okprint)
                power_pcov = np.zeros((power_k.size,power_k.size))
                listsets = ["4096_otherphase_256","4096_adaphase_256","4096_furphase_256"]
                nsim_control = 12288
                for iset in listsets:
                    for isim in range(1, 4097):
                        current_file = file_path("power", mainpath, iset, isim, noutput)
                        if (okprint) :
                            print current_file
                        if (powertype == "power"):
                            dummy, power_p, dummy = read_power(current_file)
                        elif (powertype == "renormalized"):
                            dummy, power_p = renormalized_power(mainpath, iset, isim, noutput, growth_a, growth_dplus)
                        elif (powertype == "corrected"):
                            dummy, power_p = corrected_power(mainpath, iset, aexp, isim, noutput, growth_a, growth_dplus)
                        elif (powertype == "nyquist"):
                            dummy, power_p = nyquist_power(mainpath, iset, isim, noutput, aexp, growth_a, growth_dplus)
                        elif (powertype == "mcorrected"):
                            dummy, power_p = mass_corrected_power(mainpath, iset, isim, noutput, aexp, growth_a, growth_dplus)
                        for ik in range(0, power_k.size):
                            for jk in range(0, power_k.size):
                                power_pcov[ik][jk] += (power_p[ik]-power_pmean[ik])*(power_p[jk]-power_pmean[jk])
                        nsim+=1
                power_pcov /= float(nsim-1)
            
            else:
                power_k, power_pmean, power_psigma = mean_power_all_1024(powertype, mainpath, noutput, aexp, growth_a, growth_dplus, okprint)
                power_pcov = np.zeros((power_k.size,power_k.size))
                listsets = ["64_adaphase_1024","64_curiephase_1024"]
                nsim_control = 96
                for iset in listsets:
                    if (iset=="64_adaphase_1024"):
                        isimmax = 65
                    else:
                        isimmax = 33
                    for isim in range(1, isimmax):
                        current_file = file_path("power", mainpath, iset, isim, noutput)
                        if (okprint) :
                            print current_file
                        if (powertype == "power"):
                            dummy, power_p, dummy = read_power(current_file)
                        elif (powertype == "renormalized"):
                            dummy, power_p = renormalized_power(mainpath, iset, isim, noutput, growth_a, growth_dplus)
                        elif (powertype == "corrected"):
                            dummy, power_p = corrected_power(mainpath, iset, aexp, isim, noutput, growth_a, growth_dplus)
                        elif (powertype == "nyquist"):
                            dummy, power_p = nyquist_power(mainpath, iset, isim, noutput, aexp, growth_a, growth_dplus)
                        elif (powertype == "mcorrected"):
                            dummy, power_p = mass_corrected_power(mainpath, iset, isim, noutput, aexp, growth_a, growth_dplus)
                        for ik in range(0, power_k.size):
                            for jk in range(0, power_k.size):
                                power_pcov[ik][jk] += (power_p[ik]-power_pmean[ik])*(power_p[jk]-power_pmean[jk])
                        nsim+=1
                power_pcov /= float(nsim-1)

                if (nsim!=nsim_control):
                    print "total number of simulations not corresponding"
                    sys.exit()
            
            fltformat="%-.12e"
            f = open(fname, "w")
            for i in range(0, power_k.size):
                for j in range(0, power_k.size):
                    f.write(str(fltformat%power_pcov[i][j])+" ")
                f.write("\n")
            f.close()
    else:
        power_k, power_pmean, power_psigma = mean_power(powertype, mainpath, simset, isimmin, isimmax, noutput, aexp, growth_a, growth_dplus, okprint)
        power_pcov = np.zeros((power_k.size,power_k.size))
        nsim=0
        for isim in range(isimmin, isimmax):
            true_simset,true_isim = sim_iterator(simset, isim)
            current_file = file_path("power", mainpath, true_simset, true_isim, noutput)
            if (okprint) :
                print current_file
            if (powertype == "power"):
                dummy, power_p, dummy = read_power(current_file)
            elif (powertype == "renormalized"):
                dummy, power_p = renormalized_power(mainpath, true_simset, true_isim, noutput, growth_a, growth_dplus)
            elif (powertype == "corrected"):
                dummy, power_p = corrected_power(mainpath, true_simset, aexp, true_isim, noutput, growth_a, growth_dplus)
            elif (powertype == "nyquist"):
                dummy, power_p = nyquist_power(mainpath, true_simset, true_isim, noutput, aexp, growth_a, growth_dplus)
            elif (powertype == "mcorrected"):
                dummy, power_p = mass_corrected_power(mainpath, true_simset, true_isim, noutput, aexp, growth_a, growth_dplus)
            for ik in range(0, power_k.size):
                for jk in range(0, power_k.size):
                    power_pcov[ik][jk] += (power_p[ik]-power_pmean[ik])*(power_p[jk]-power_pmean[jk])
            nsim+=1
        #print n_sim, nsim
        power_pcov /= float(nsim-1)

    return power_k, power_pmean, power_psigma, power_pcov
# ---------------------------------------------------------------------------- #



# -------------------- KMAX LIMITED COVARIANCE POWER ------------------------- #
def cov_power_kmax(power_type, kmax, power_k, power_pmean, power_psigma, mainpath = "", simset = "", isimmin = 1, isimmax = 2, noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), sampling=0, okprint = True):
    nsim = 0
    
    idx=(power_k < kmax)
    power_k_kmax = power_k[idx]

    # undersampling
    if(sampling==-1):
        Nk = np.zeros(power_k.size)
        for ik in range(0,power_k.size):
            Nk[ik] = power_k[ik] * power_k[ik]
        power_k_und = np.zeros(power_k.size/2)
        for index in range(0,power_k.size/2):
            power_k_und[index]=0.5*(power_k[index*2]+power_k[index*2+1])
        idx = (power_k_und < kmax)
        power_k_kmax = power_k_und[idx]
        power_pmean_samp = np.zeros(power_pmean.size/2)
        for index in range(0,power_pmean.size/2):
            power_pmean_samp[index]=(Nk[index*2]*power_pmean[index*2]+Nk[index*2+1]*power_pmean[index*2+1])/(power_k_und[index]*power_k_und[index]*2.)

    #oversampling (warning: this generate divergencies in covariance inversion!!)
    elif(sampling==1):
        power_k_over = np.zeros(power_k_kmax.size*2)
        power_pmean_samp = np.zeros(power_pmean.size*2)
        for index in range(0,power_k_kmax.size*2-1):
            if(index%2==0):
                power_k_over[index]=power_k_kmax[index/2]
            else:
                power_k_over[index]=(power_k_kmax[index/2]+power_k_kmax[index/2+1])/2.
        power_k_kmax = power_k_over
        for index in range(0,power_pmean.size*2-1):
            if(index%2==0):
                power_pmean_samp[index]=power_pmean[index/2]
            else:
                power_pmean_samp[index]=(power_pmean[index/2]+power_pmean[index/2+1])/2.
    else:
        power_pmean_samp=power_pmean

    power_pcov = np.zeros((power_k_kmax.size,power_k_kmax.size))
    for isim in range(isimmin, isimmax):
        # plain sets or composite sets
        if (simset == "all_256"):
            listsets = ["4096_otherphase_256","4096_adaphase_256","4096_furphase_256"]
            if (isim<=4096):
                iset = listsets[0]
                isimu = isim
            elif (isim <= 8192):
                iset = listsets[1]
                isimu = isim - 4096
            else:
                iset = listsets[2]
                isimu = isim - 8192
        elif (simset == "all_1024"):
            listsets = ["64_adaphase_1024","64_curiephase_1024"]
            if (isim<=64):
                iset = listsets[0]
                isimu = isim
            else:
                iset = listsets[1]
                isimu = isim - 64
        else:
            iset = simset
            isimu = isim
        
        # true simulations or randomly generated power spectra
        if (simset=="64_adaphase_1024" and isim>64):
            power_k, dummy = nyquist_power(mainpath, simset, 1, noutput, aexp, growth_a, growth_dplus)
            power_p = np.zeros(power_k_kmax.size)
            for ind in range(0,power_k_kmax.size):
                power_p[ind] = np.random.normal(power_pmean[ind],power_psigma[ind],1)
        else:
            current_file = file_path("power", mainpath, iset, isimu, noutput)
            if (okprint) :
                print current_file
            if (power_type == "mcorrected"):
                power_k, power_p = mass_corrected_power(mainpath, iset, isimu, noutput, aexp, growth_a, growth_dplus)
            else:
                power_k, power_p = nyquist_power(mainpath, iset, isimu, noutput, aexp, growth_a, growth_dplus)

        # deal with under or over sampling
        if(sampling==-1):
            power_p_und = np.zeros(power_p.size/2)
            for index in range(0,power_p.size/2):
                power_p_und[index]=(power_k[index*2]*power_k[index*2]*power_p[index*2]+power_k[index*2+1]*power_k[index*2+1]*power_p[index*2+1])/(power_k_und[index]*power_k_und[index])
            power_p=power_p_und
        elif(sampling==1):
            power_p_over = np.zeros(power_p.size*2)
            for index in range(0,power_p.size*2-1):
                if(index%2==0):
                    power_p_over[index]=power_p[index/2]
                else:
                    power_p_over[index]=(power_p[index/2]+power_p[index/2+1])/2.
            power_p = power_p_over
        
        for ik in range(0, power_k_kmax.size):
            for jk in range(0, power_k_kmax.size):
                power_pcov[ik][jk] += (power_p[ik]-power_pmean_samp[ik])*(power_p[jk]-power_pmean_samp[jk])
        nsim+=1
    power_pcov /= float(nsim-1)
    return power_k_kmax, power_pcov
# ---------------------------------------------------------------------------- #



# -------------------- CORRELATION COEFFICIENT POWER ------------------------- #
def corr_coeff(powertype = "power", mainpath = "", simset = "", isimmin = 1, isimmax = 2, noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), okprint = False):

    if (simset=="all_256" and (isimmax-isimmin)==12288):
        power_k, power_pmean, power_psigma = mean_power_all(powertype, mainpath, noutput, aexp, growth_a, growth_dplus, okprint)
    elif(simset=="all_1024" and (isimmax-isimmin)==96):
        power_k, power_pmean, power_psigma = mean_power_all_1024(powertype, mainpath, noutput, aexp, growth_a, growth_dplus, okprint)
    else:
        power_k, power_pmean, power_psigma = mean_power(powertype, mainpath, simset, isimmin, isimmax, noutput, aexp, growth_a, growth_dplus, okprint)
        
    backup_cov(powertype, mainpath, simset, noutput,aexp,growth_a,growth_dplus)
    fname = "cov_"+simset+"_"+powertype+"_"+str("%05d"%noutput)+".txt"
    power_pcov = np.loadtxt(fname,unpack=True)
    corr_coeff = np.zeros((power_k.size,power_k.size))
    for ik in range(0, power_k.size):
        for jk in range(0, power_k.size):
            corr_coeff[ik][jk] = power_pcov[ik][jk]/(sqrt(power_pcov[ik][ik])*sqrt(power_pcov[jk][jk]))#/(power_psigma[ik]*power_psigma[jk])
            
    return corr_coeff
# ---------------------------------------------------------------------------- #



# -------------------- CORRELATION COEFFICIENT POWER FROM BACKUP ------------------------- #
def corr_coeff_b(powertype = "power", mainpath = "", simset = "", isimmin = 1, isimmax = 2, noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), filename = "", okprint = False):
    if (simset=="all_256" and (isimmax-isimmin)==12288):
        power_k, power_pmean, power_psigma = mean_power_all(powertype, mainpath, noutput, aexp, growth_a, growth_dplus, okprint)
    elif (simset=="all_1024" and (isimmax-isimmin)==96):
        power_k, power_pmean, power_psigma = mean_power_all_1024(powertype, mainpath, noutput, aexp, growth_a, growth_dplus, okprint)
    else:
        power_k, power_pmean, power_psigma = mean_power(powertype, mainpath, simset, isimmin, isimmax, noutput, aexp, growth_a, growth_dplus, okprint)
            
    corr_coeff = np.zeros((power_k.size,power_k.size))
    power_pcov = np.zeros((power_k.size,power_k.size))
    power_pcov = np.loadtxt(filename, unpack=True)
    
    for ik in range(0, power_k.size):
        for jk in range(0, power_k.size):
            corr_coeff[ik][jk] = power_pcov[ik][jk]/(power_psigma[ik]*power_psigma[jk])
    
    return corr_coeff
# ---------------------------------------------------------------------------- #



# --------------------------- SAMPLE COVARIANCE VARIANCE --------------------------- #
def cov_variance(powertype = "power", mainpath = "", noutput = 1 , aexp = 1., growth_a = np.zeros(0), growth_dplus = np.zeros(0), list_nr = [10,50,100,500,700,1000,3000,5000,6000], okprint=False):

    simset = "all_256"
    trace_sigma2=np.zeros(len(list_nr))
    i=0 #index on list_nr
    if (powertype=="power"):
        power_k, dummy, dummy = read_power(file_path("power", mainpath, "4096_adaphase_256", 1, noutput))
    elif (powertype=="nyquist" or powertype=="mcorrected"):
        power_k,dummy = nyquist_power(mainpath,"4096_adaphase_256",1,noutput,aexp,growth_a,growth_dplus)
    elif (powertype=="corrected"):
        power_k, dummy = corrected_power(mainpath,"4096_adaphase_256",aexp,1,noutput,growth_a,growth_dplus)
    else:
        print "power type not supported"

    for nr in list_nr:

        nsub = 12288/nr
        nbin = power_k.size
        power_pmean = np.zeros((nsub,nbin))
        power_pvar = np.zeros((nsub,nbin))
        nsim = 0
        totsim = 12288 - int(math.fmod(12288,nr))
        
        
        print nr,totsim
        subset=0 #index of the subset
        for isim in range(1, totsim+1):
            true_simset, true_isim = sim_iterator("all_256",isim)
            current_file = file_path("power", mainpath, true_simset, true_isim, noutput)
            if (okprint) :
                print current_file
            if (powertype=="power"):
                dummy, power_p, dummy = read_power(current_file)
            elif (powertype=="nyquist"):
                dummy, power_p = nyquist_power(mainpath, true_simset, true_isim, noutput, aexp, growth_a, growth_dplus)
            elif (powertype=="corrected"):
                dummy, power_p = corrected_power(mainpath,true_simset,aexp,true_isim,noutput,growth_a,growth_dplus)
            elif (powertype=="mcorrected"):
                dummy, power_p = mass_corrected_power(mainpath, true_simset, true_isim, noutput, aexp, growth_a, growth_dplus)
            else:
                print "power type not supported"
            for ik in range(0,nbin):
                power_pmean[subset][ik]+=power_p[ik]/float(nr)
            if (math.fmod(isim,nr)==0 and isim>0):
                subset+=1
            nsim+=1

        print nsim, nr

        subset=0
        for isim in range(1, totsim+1):
            true_simset, true_isim = sim_iterator("all_256",isim)
            current_file = file_path("power", mainpath, true_simset, true_isim, noutput)
            if (okprint) :
                print current_file
            if (powertype=="power"):
                dummy, power_p, dummy = read_power(current_file)
            elif (powertype=="nyquist"):
                dummy, power_p = nyquist_power(mainpath, true_simset, true_isim, noutput, aexp, growth_a, growth_dplus)
            elif (powertype=="corrected"):
                dummy, power_p = corrected_power(mainpath,true_simset,aexp,true_isim,noutput,growth_a,growth_dplus)
            elif (powertype=="mcorrected"):
                dummy, power_p = mass_corrected_power(mainpath, true_simset, true_isim, noutput, aexp, growth_a, growth_dplus)
            else:
                print "power type not supported"
            
            for ik in range(0,nbin):
                power_pvar[subset][ik] += (power_p[ik]-power_pmean[subset][ik])*(power_p[ik]-power_pmean[subset][ik])/float(nr-1)
            if (math.fmod(isim,nr)==0 and isim > 0):
                subset+=1

        #power_pvar /= float(nr -1)

        #dummy,power_pmean_all,power_psigma_all=mean_power(powertype, mainpath, "all_256", 1, totsim+1, noutput, aexp, growth_a, growth_dplus)
        #var_all = power_psigma_all*power_psigma_all

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

    if (powertype=="power"):
        power_k, dummy, dummy = read_power(file_path("power", mainpath, "4096_adaphase_256", 1, noutput))
    elif (powertype=="nyquist" or powertype=="mcorrected"):
        power_k,dummy = nyquist_power(mainpath,"4096_adaphase_256",1,noutput,aexp,growth_a,growth_dplus)
    elif (powertype=="corrected"):
        power_k, dummy = corrected_power(mainpath,"4096_adaphase_256",aexp,1,noutput,growth_a,growth_dplus)
    else:
        print "power type not supported"

    for nr in list_nr:
        nsub = 12288/nr
        nbin = power_k.size
        var_inv = np.zeros((nsub,nbin))
        nsim = 0
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
        
        #print "all begin"
        #filename1="tmp/inv_cov_"+powertype+"_"+str("%05d"%totsim)+"_all.txt"
        #if(os.path.isfile(filename1)):
        #    cov_inv_all=np.loadtxt(filename, unpack=True)
        #else:
        #    k_cov_all,dummy,dummy,cov_all=cov_power(powertype,mainpath,simset,1,totsim+1,noutput,aexp,growth_a,growth_dplus)
        #    cov_inv_all=linalg.inv(cov_all)
        #    cov_inv_all=float(totsim-nbin-2)*cov_inv_all/float(totsim-1)
        #    fltformat="%-.12e"
        #    f = open(filename1, "w")
        #    for ik in range(0, nbin):
        #        for jk in range(0, nbin):
        #            f.write(str(fltformat%cov_inv_all[ik][jk])+" ")
        #        f.write("\n")
        #    f.close()
        #print "all end"
        #for ik in range(0,nbin):
        #    var_all[ik]=cov_inv_all[ik][ik]

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
#        var_inv = np.zeros((nsub,nbin))
        nsim = 0
        totsim = 12288 - int(math.fmod(12288,nr))
        
        print nr,totsim

        var_inv_mean = np.zeros(nbin)
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


            for ik in range(0,nbin):
                var_inv_mean[ik]+=cov_inv[ik][ik]
        var_inv_mean /= float(nsub)

        trace=0.
        for ik in range(0,nbin):
            trace+=var_inv_mean[ik]
        print trace
        bias[i]=(trace-trace_all)/trace_all
        i+=1
    return list_nr, bias
# ---------------------------------------------------------------------------- #



# -------------------------- COVARIANCE CONVERGENCE -------------------------- #
def cov_convergence(powertype = "power", mainpath = "", noutput = 1 , aexp = 1., growth_a = np.zeros(0), growth_dplus = np.zeros(0), kref1=0.05, kref2=0.4, list_nr = [10,50,100,500,700,1000,3000,5000,6000], okprint=False):
    #power_k, mean_all, sigma_all, cov_all = cov_power(powertype, mainpath, "all_256", 1, 12288, noutput, aexp, growth_a, growth_dplus)
    power_k, mean_all, sigma_all = mean_power_all(powertype, mainpath, noutput)
    ik1 = 0
    ik2 = 0
    for k in range(0,power_k.size):
        if (power_k[k] < kref1 + 0.0024 and power_k[k] > kref1 - 0.0024):
            ik1 = k
        if (power_k[k] < kref2 + 0.0024 and power_k[k] > kref2 - 0.0024):
            ik2 = k
    if (ik1==0 or ik2==0):
        print "kref not found"
    #var_all = sigma_all[ik] * sigma_all[ik]
    #power_pvar = np.zeros(len(list_nr))
    #power_pmean = np.zeros(len(list_nr))
    sigma2=np.zeros(len(list_nr))

    i=0 #index of list_nr
    for nr in list_nr:
        cov_all12=0.
        power_pmean = np.zeros(power_k.size)
        power_pmean1 = np.zeros(12288/nr)
        power_pmean2 = np.zeros(12288/nr)
        power_pcov = np.zeros(12288/nr)
        nsim = 0
        totsim = 12288 - int(math.fmod(12288,nr))
        print nr,totsim
        subset=0
        #while (subset < 12288/nr):
        for isim in range(1, totsim+1):
            true_simset, true_isim = sim_iterator("all_256",isim)
            current_file = file_path("power", mainpath, simset, true_isim, noutput)
            if (okprint) :
                print current_file
            if (powertype=="power"):
                dummy, power_p, dummy = read_power(current_file)
            elif (powertype=="nyquist" or powertype=="corrected"):
                dummy, power_p = corrected_power(mainpath,simset,aexp,true_isim,noutput,growth_a,growth_dplus)
            elif (powertype=="mcorrected"):
                dummy, power_p = mass_corrected_power(mainpath, simset, true_isim, noutput, aexp, growth_a, growth_dplus)
            else:
                print "power type not supported"
            power_pmean1[subset] += power_p[ik1]
            power_pmean2[subset] += power_p[ik2]
            for ik in range(0,power_k.size):
                power_pmean[ik] += power_p[ik]
            if (math.fmod(isim,nr)==0 and isim>0):
                subset+=1
                    #print subset
            nsim+=1
        power_pmean1 /= float(nr)
        power_pmean2 /= float(nr)
        power_pmean /= float(totsim)
        print nsim, nr

        subset=0
        for isim in range(1, totsim+1):
            true_simset, true_isim = sim_iterator("all_256",isim)
            current_file = file_path("power", mainpath, simset, true_isim, noutput)
            if (okprint) :
                print current_file
            dummy, power_p, dummy = read_power(current_file)
            power_pcov[subset] += (power_p[ik1]-power_pmean1[subset])*(power_p[ik2]-power_pmean2[subset])
            cov_all12 += (power_p[ik1]-power_pmean[ik1])*(power_p[ik2]-power_pmean[ik2])
                
            if (math.fmod(isim,nr)==0 and isim > 0):
                subset+=1
        power_pcov /= float(nr -1)
        cov_all12 /= float(totsim-1)

        for isubset in range(0,12288/nr):
                sigma2[i]+=(power_pcov[isubset] - cov_all12) * (power_pcov[isubset] - cov_all12)
        sigma2[i]/=float(12288/nr * cov_all12 * cov_all12)
        i+=1
    return list_nr, sigma2
# ---------------------------------------------------------------------------- #



# -------------------------------- BACKUP COVARIANCE ------------------------------- #
def backup_cov(power_type = "nyquist", mainpath = "", simset = "", ioutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), fltformat = "%-.12e", okprint = False):
    if (simset == "4096_furphase" or simset == "4096_otherphase"):
        isimmax = 1
        noutputmax = 31
    elif (simset == "4096_furphase_512"):
        isimmax = 512
        noutputmax = 9
    elif (simset == "64_adaphase_1024" or simset == "64_curiephase_1024"):
        isimmax = 64
        noutputmax = 9
    elif (simset == "all_256"):
        isimmax = 12288
        noutputmax = 9
    elif (simset == "all_1024"):
        isimmax = 96
        noutputmax = 9
    else:
        isimmax = 4096
        noutputmax = 9
    #for ioutput in range(noutputmax, noutputmax+1):
    power_k, power_pmean, power_psigma, power_pcov = cov_power(power_type, mainpath, simset, 1, isimmax+1, ioutput, aexp, growth_a, growth_dplus, okprint)

    fname = "cov_"+simset+"_"+power_type+"_"+str("%05d"%ioutput)+".txt"
    f = open(fname, "w")
    for i in range(0, power_k.size):
        for j in range(0, power_k.size):
            f.write(str(fltformat%power_pcov[i][j])+" ")
        f.write("\n")
    f.close()
    return
# ---------------------------------------------------------------------------- #



# ----------------------------- SIGNAL TO NOISE ------------------------------ #
def signoise(power_type = "nyquist", mainpath = "", simset = "", noutput = 1, nsimmax = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), sampling = 0, unbiased = False, okprint = False):
    fname="sig_noise_"+power_type+"_"+simset+"_"+str("%05d"%noutput)+".txt"
    if(os.path.isfile(fname) and simset=="all_256" and nsimmax==12289):
        kmax, sig_noise = np.loadtxt(fname,unpack=True)
    else:
    
        if(simset=="64_adaphase_1024"):
            isimmax=64
            sim_set = simset
        elif(simset=="64_curiephase_1024"):
            isimmax=24
            sim_set = simset
        elif(simset=="4096_furphase_512"):
            isimmax=512
            sim_set = simset
        elif(simset=="all_256"):
            isimmax=12288
            sim_set = "4096_furphase_256"
        elif(simset=="all_1024"):
            isimmax=96
            sim_set = "64_adaphase_1024"
        else:
            isimmax=4096
            sim_set = simset

        nmax = isimmax
        step = 1
    
#        if (power_type == "mcorrected"):
#            power_k, power_p = mass_corrected_power(mainpath, sim_set, 1, noutput, aexp, growth_a, growth_dplus)
#        else:
#            power_k, power_p = nyquist_power(mainpath, sim_set, 1, noutput, aexp, growth_a, growth_dplus)

#        if (sampling==-1):
#            power_p_und = np.zeros(power_p.size/2)
#            power_k_samp = np.zeros(power_k.size/2)
#            for index in range(0,power_p.size/2):
#                power_k_samp[index]=0.5*(power_k[index*2]+power_k[index*2+1])
#            for index in range(0,power_p.size/2):
#                power_p_und[index]=(power_k[index*2]*power_k[index*2]*power_p[index*2]+power_k[index*2+1]*power_k[index*2+1]*power_p[index*2+1])/(power_k_samp[index]*power_k_samp[index])
#            power_p = power_p_und
#            step = 2
        #nmax = 2*isimmax
#        elif (sampling==1):
#            power_p_over = np.zeros(power_p.size*2-1)
#            power_k_samp = np.zeros(power_k.size*2-1)
#            for index in range(0,power_p.size*2-1):
#                if(index%2==0):
#                    power_p_over[index]=power_p[index/2]
#                    power_k_samp[index]=power_k[index/2]
#                else:
#                    power_p_over[index]=(power_p[index/2]+power_p[index/2+1])/2.
#                    power_k_samp[index]=(power_k[index/2]+power_k[index/2+1])/2.
#            power_p = power_p_over
#            step = 8
#            nmax = isimmax/2
#        else:
#            nmax=nsimmax-1
#            power_k_samp = power_k
    
        power_k, power_pmean, power_psigma, power_pcov = cov_power(power_type, mainpath,simset, 1, isimmax+1, noutput, aexp, growth_a, growth_dplus)

        power_k, power_pmean, dummy = mean_power_all("mcorrected",mainpath,noutput,aexp,growth_a,growth_dplus)

        kmin=power_k[0]
        num_iter=min(power_k.size-1,nmax-2)
        sig_noise = np.zeros(num_iter/step)
        kmax = np.zeros(num_iter/step)
        print kmax.size, num_iter
        for ikmax in range(0,num_iter/step):
            ikk=ikmax*step+1
            kmax[ikmax]=power_k[ikk]
            idx=(power_k < kmax[ikmax])
            power_p_new=power_pmean[idx]
            #power_kk,power_pcov=cov_power_kmax(power_type, power_k[ikk], power_k, power_pmean, power_psigma, mainpath, simset, 1, nsimmax, noutput, aexp, growth_a, growth_dplus, sampling, okprint)
            power_kk = power_k[idx]
            power_pcov_kmax=power_pcov[0:ikk,0:ikk]
            cov_inv=linalg.inv(power_pcov_kmax)
            if (unbiased):
                cov_inv= float(nmax-num_iter-2)*cov_inv/float(nmax-1)
            signoise2=0.
            for ik in range(0,power_kk.size):
                for jk in range(0,power_kk.size):
                    signoise2+=power_p_new[ik]*cov_inv[ik][jk]*power_p_new[jk]
            sig_noise[ikmax]=sqrt(signoise2)
            #print kmax[ikmax],sig_noise[ikmax]

        fltformat = "%-.12e"
        f = open(fname, "w")
        for i in range(0, kmax.size):
            f.write(str(fltformat%kmax[i])+" "+str(fltformat%sig_noise[i])+"\n")
        f.close()

    return kmax, sig_noise
# --------------------------------------------------------------------------- #



# ----------------------------- SIGNAL TO NOISE AT A GIVEN K ------------------------------ #
def signoise_k(power_type = "nyquist", mainpath = "", simset = "", noutput = 1, nsimmax = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), kmax=0.4, unbiased = False, okprint = False):
    
    if(simset=="64_adaphase_1024"):
        isimmax=64
        sim_set = simset
    elif(simset=="64_curiephase_1024"):
        isimmax=24
        sim_set = simset
    elif(simset=="4096_furphase_512"):
        isimmax=512
        sim_set = simset
    elif(simset=="all_256"):
        isimmax=12288
        sim_set = "4096_furphase_256"
    elif(simset=="all_1024"):
        isimmax=96
        sim_set = "64_adaphase_1024"
    else:
        isimmax=4096
        sim_set = simset

    nmax = nsimmax-1
    step = 4
    
    if (power_type == "mcorrected"):
        power_k, power_p = mass_corrected_power(mainpath, sim_set, 1, noutput, aexp, growth_a, growth_dplus)
    else:
        power_k, power_p = nyquist_power(mainpath, sim_set, 1, noutput, aexp, growth_a, growth_dplus)
    
    if (simset=="all_256" and nsimmax==12289):
        power_k, power_pmean, power_psigma = mean_power_all(power_type, mainpath, noutput, aexp, growth_a, growth_dplus)
    elif (simset=="all_1024"):
        power_k, power_pmean, power_psigma = mean_power_all_1024("nyquist", mainpath, noutput, aexp, growth_a, growth_dplus)
    else:
        power_k, power_pmean, power_psigma = mean_power(power_type, mainpath, simset, 1, nsimmax, noutput, aexp, growth_a, growth_dplus)
    
    num_iter=min(power_k.size,nmax-2)
    idx=(power_k < kmax)
    power_p_new=power_p[idx]
    power_kk,power_pcov=cov_power_kmax(power_type, kmax, power_k, power_pmean, power_psigma, mainpath, simset, 1, nsimmax, noutput, aexp, growth_a, growth_dplus, 0, okprint)
    cov_inv=linalg.inv(power_pcov)
    if (unbiased):
        cov_inv= float(nmax-num_iter-2)*cov_inv/float(nmax-1)
    signoise2=0.
    for ik in range(0,power_kk.size):
        for jk in range(0,power_kk.size):
            signoise2+=power_p_new[ik]*cov_inv[ik][jk]*power_p_new[jk]
    sig_noise=sqrt(signoise2)

    return kmax, sig_noise
# --------------------------------------------------------------------------- #



# ------------- BACKUP CORRELATION COEFFICIENT ------------------------------- #
def backup_corr(powertype="power", mainpath = "", simset = "", ioutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), fltformat = "%-.12e", cov_backup = False):
    if (simset == "4096_furphase" or simset == "4096_otherphase"):
        isimmax = 1
        noutputmax = 31
        sim_set = simset
    elif (simset == "4096_furphase_512"):
        isimmax = 512
        noutputmax = 9
        sim_set = simset
    elif (simset == "64_adaphase_1024"):
        isimmax = 64
        noutputmax = 9
        sim_set = simset
    elif (simset == "64_curiephase_1024"):
        isimmax = 16
        noutputmax = 9
        sim_set = simset
    elif (simset == "all_256"):
        isimmax = 12288
        noutputmax = 9
        sim_set = "4096_furphase_256"
    elif (simset == "all_1024"):
        isimmax = 96
        noutputmax = 9
        sim_set = "64_adaphase_1024"
    else:
        isimmax = 4096
        noutputmax = 9
        sim_set = simset
    #for ioutput in range(noutputmax-1,noutputmax):
    fname_cov="cov_"+simset+"_"+powertype+"_"+str("%05d"%ioutput)+".txt"
    fname_corr="corr_coeff_"+powertype+"_"+simset+"_"+str("%05d"%ioutput)+".txt"
    if (cov_backup):
        corr_coeffi = corr_coeff_b(powertype, mainpath, simset, 1, isimmax+1, ioutput, aexp, growth_a, growth_dplus, fname_cov)
    else:
        corr_coeffi = corr_coeff(powertype, mainpath, simset, 1, isimmax+1, ioutput, aexp, growth_a, growth_dplus)
    
    if (powertype == "nyquist"):
        power_k, dummy = nyquist_power(mainpath, sim_set, 1, ioutput, aexp, growth_a, growth_dplus)
    elif(powertype == "mcorrected"):
        power_k, dummy = mass_corrected_power(mainpath, sim_set, 1, ioutput, aexp, growth_a, growth_dplus)
    else:
        power_k, dummy, dummy = read_power(file_path("power", mainpath, sim_set, 1, ioutput))

    f = open(fname_corr, "w")
    for i in range(0, power_k.size):
        for j in range(0, power_k.size):
            #print i, j, corr_coeffi[i][j]
            f.write(str(fltformat%corr_coeffi[i][j])+" ")
        f.write("\n")
    f.close()
    return
# ---------------------------------------------------------------------------- #
