#!/usr/bin/env python
# power_stats.py - Vincent Reverdy (vince.rev@gmail.com) and Linda Blot (linda.blot@obspm.fr) - 2013
# ---------------------------------- IMPORT ---------------------------------- #
import math
import glob
import os
import sys
import numpy as np
import scipy.interpolate as itl
from scipy import stats
from lmfit import minimize, Parameters
from utilities import *
from read_files import *
from power_types import *
import random
# ---------------------------------------------------------------------------- #



# -------------------------------- MEAN POWER -------------------------------- #
def mean_power(powertype = "power", mainpath = "", simset = "", isimmin = 1, isimmax = 2, noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), okprint = False):
    
    nsim = isimmax-isimmin
    if (simset=="all_256" and nsim==12288):
        power_k, power_pmean, power_psigma = mean_power_all(powertype,mainpath,noutput,aexp,growth_a,growth_dplus,okprint)
    elif (simset=="all_1024" and nsim==96):
        power_k, power_pmean, power_psigma = mean_power_all1024(powertype,mainpath,noutput,aexp,growth_a,growth_dplus,okprint)
    else:
        true_simset, true_isimmin = sim_iterator(simset, isimmin)
        power_k, power_p = power_spectrum(powertype,mainpath,true_simset,true_isimmin,noutput,aexp,growth_a,growth_dplus)
        power_pmean = np.zeros(power_k.size)
        power_psigma = np.zeros(power_k.size)

        for isim in range(isimmin, isimmax):
            true_simset,true_isim = sim_iterator(simset, isim)
            if (okprint) :
                current_file = file_path("power", mainpath, true_simset, true_isim, noutput)
                print current_file
            dummy, power_p = power_spectrum(powertype,mainpath,true_simset,true_isim,noutput,aexp,growth_a,growth_dplus,okprint)
            power_pmean += power_p

        power_pmean /= float(nsim)

        if (nsim > 1):
            for isim in range(isimmin, isimmax):
                true_simset,true_isim = sim_iterator(simset, isim)
                if (okprint) :
                    current_file = file_path("power", mainpath, true_simset, true_isim, noutput)
                    print current_file
                dummy, power_p = power_spectrum(powertype,mainpath,true_simset,true_isim,noutput,aexp,growth_a,growth_dplus,okprint)
                power_psigma += (power_p-power_pmean)*(power_p-power_pmean)
            power_psigma /= float(nsim-1)
            power_psigma = np.sqrt(power_psigma)

    return power_k, power_pmean, power_psigma
# ---------------------------------------------------------------------------- #



# ------------------------------- MEAN POWER ON 12288 ------------------------ #
def mean_power_all(powertype = "power", mainpath = "", noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), okprint = False):
    
    fname = "mean_"+powertype+"_256_"+str("%05d"%noutput)+".txt"
    simset = "all_256"
    if(os.path.isfile(fname)):
        power_k, power_pmean, power_psigma = np.loadtxt(fname,unpack=True)
    else:
        nsim = 12288
        true_simset, true_isimmin = sim_iterator(simset, 1)
        power_k, power_p = power_spectrum(powertype,mainpath,true_simset,true_isimmin,noutput,aexp,growth_a,growth_dplus,okprint)
        power_pmean = np.zeros(power_k.size)
        power_psigma = np.zeros(power_k.size)
        
        for isim in range(1,nsim+1):
            true_simset,true_isim = sim_iterator(simset, isim)
            if (okprint) :
                current_file = file_path("power", mainpath, true_simset, true_isim, noutput)
                print current_file
            dummy, power_p = power_spectrum(powertype,mainpath,true_simset,true_isim,noutput,aexp,growth_a,growth_dplus,okprint)
            power_pmean += power_p
            
        power_pmean /= float(nsim)

        for isim in range(1,nsim+1):
            true_simset,true_isim = sim_iterator(simset, isim)
            if (okprint) :
                current_file = file_path("power", mainpath, true_simset, true_isim, noutput)
                print current_file
            dummy, power_p = power_spectrum(powertype,mainpath,true_simset,true_isim,noutput,aexp,growth_a,growth_dplus,okprint)
            power_psigma += (power_p-power_pmean)*(power_p-power_pmean)
        power_psigma /= float(nsim-1)
        power_psigma = np.sqrt(power_psigma)

        f = open(fname, "w")
        for i in range(0, power_k.size):
            f.write(str("%-.12e"%power_k[i])+" "+str("%-.12e"%power_pmean[i])+" "+str("%-.12e"%power_psigma[i])+"\n")
        f.close()
            
    return power_k, power_pmean, power_psigma
# ---------------------------------------------------------------------------- #



# ------------------------------- MEAN POWER ON ALL 1024 ------------------------ #
def mean_power_all_1024(powertype = "power", mainpath = "", noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), okprint = False):
    
    fname = "mean_"+powertype+"_1024_"+str("%05d"%noutput)+".txt"
    simset = "all_1024"
    if(os.path.isfile(fname)):
        power_k, power_pmean, power_psigma = np.loadtxt(fname,unpack=True)
    else:
        nsim = 96
        true_simset, true_isimmin = sim_iterator(simset, 1)
        power_k, power_p = power_spectrum(powertype,mainpath,true_simset,true_isimmin,noutput,aexp,growth_a,growth_dplus)
        power_pmean = np.zeros(power_k.size)
        power_psigma = np.zeros(power_k.size)
        
        for isim in range(1,nsim+1):
            true_simset,true_isim = sim_iterator(simset, isim)
            if (okprint) :
                current_file = file_path("power", mainpath, true_simset, true_isim, noutput)
                print current_file
            dummy, power_p = power_spectrum(powertype,mainpath,true_simset,true_isim,noutput,aexp,growth_a,growth_dplus,okprint)
            power_pmean += power_p

        power_pmean /= float(nsim)

        for isim in range(1,nsim+1):
            true_simset,true_isim = sim_iterator(simset, isim)
            if (okprint) :
                current_file = file_path("power", mainpath, true_simset, true_isim, noutput)
                print current_file
            dummy, power_p = power_spectrum(powertype,mainpath,true_simset,true_isim,noutput,aexp,growth_a,growth_dplus,okprint)
            power_psigma += (power_p-power_pmean)*(power_p-power_pmean)
        power_psigma /= float(nsim-1)
        power_psigma = np.sqrt(power_psigma)
        
        f = open(fname, "w")
        for i in range(0, power_k.size):
            f.write(str("%-.12e"%power_k[i])+" "+str("%-.12e"%power_pmean[i])+" "+str("%-.12e"%power_psigma[i])+"\n")
        f.close()
            
    return power_k, power_pmean, power_psigma
# ---------------------------------------------------------------------------- #



# --------------------- PDF OF POWER SPECTRA --------------------------- #

def distrib_power(powertype = "power", mainpath = "", simset = "", isimmin = 1, isimmax = 2, noutput = 1, nbin = 50, kref = 0.2, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), okprint = False):
    
    nsim = isimmax - isimmin

    power_values = np.zeros(nsim)
        
    for isim in range(1, nsim+1):
        true_simset, true_isim = sim_iterator(simset,isim)
        if (okprint) :
            current_file = file_path("power", mainpath, true_simset, true_isim, noutput)
            print current_file
        power_k, power_p = power_spectrum(powertype,mainpath,true_simset,true_isim,noutput,aexp,growth_a,growth_dplus)
        
        for ik in range(0,power_k.size):
            if (power_k[ik]-kref < 0.0048):
                power_values[isim-1] = power_p[ik]
                control = 1
        if (control == 0):
            print "k out of range"

    pmin = min(power_values)
    pmax = max(power_values)
    binstep = (pmax - pmin) / float(nbin)
    npower_bin = np.zeros(nbin)
    bincenter = np.zeros(nbin)
    for ibin in range(0,nbin):
        binmin = pmin + ibin*binstep
        binmax = binmin + binstep
        bincenter[ibin] = (binmin + binmax) / 2.
        for isimu in range(0,nsim):
            if (power_values[isimu] >= binmin and power_values[isimu] < binmax):
                npower_bin[ibin]+=1
    npower_bin/=float(nsim)

    return bincenter, npower_bin

# ---------------------------------------------------------------------------- #



# ------------------ HIGH MOMENTS OF SPECTRA PDF ---------------------- #
def high_moments(powertype = "power", mainpath = "", simset = "", isimmin = 1, isimmax = 2, noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), unbiased = True, okprint = False):

    nsim = 0
    if(unbiased):
        bias="unbiased"
    else:
        bias="biased"

    fname="high_moments_"+bias+"_"+powertype+"_"+simset+"_"+str("%05d"%noutput)+".txt"
    if(os.path.isfile(fname)):
        power_k, power_skew, power_kurt = np.loadtxt(fname,unpack=True)
    else:

        nsim = isimmax-isimmin

        if (simset=="all_256" and nsim==12288):
            power_k, power_pmean, power_psigma = mean_power_all(powertype, mainpath, noutput, aexp, growth_a, growth_dplus, False)
        
            power_skew = np.zeros(power_k.size)
            power_kurt = np.zeros(power_k.size)
    
            for isim in range(1, nsim+1):
                true_simset, true_isim = sim_iterator(simset,isim)
                if (okprint) :
                    current_file = file_path("power", mainpath, true_simset, true_isim, noutput)
                    print current_file
                power_k, power_p = power_spectrum(powertype,mainpath,true_simset,true_isim,noutput,aexp,growth_a,growth_dplus)
        
                power_skew += (power_p-power_pmean)*(power_p-power_pmean)*(power_p-power_pmean)
                power_kurt += (power_p-power_pmean)*(power_p-power_pmean)*(power_p-power_pmean)*(power_p-power_pmean)

        else:
            power_k, power_pmean, power_psigma = mean_power(powertype, mainpath, simset, isimmin, isimmax, noutput, aexp, growth_a, growth_dplus, okprint)
    
            power_skew = np.zeros(power_k.size)
            power_kurt = np.zeros(power_k.size)
    
            for isim in range(isimmin, isimmax+1):
                true_simset, true_isim = sim_iterator(simset,isim)
                if (okprint) :
                    current_file = file_path("power", mainpath, true_simset, true_isim, noutput)
                    print current_file
                power_k, power_p = power_spectrum(powertype,mainpath,true_simset,true_isim,noutput,aexp,growth_a,growth_dplus)
                power_skew += (power_p-power_pmean)*(power_p-power_pmean)*(power_p-power_pmean)
                power_kurt += (power_p-power_pmean)*(power_p-power_pmean)*(power_p-power_pmean)*(power_p-power_pmean)


        if (unbiased):
            power_skew *= float(nsim)/(float(nsim-1)*float(nsim-2))
            power_skew /= power_psigma * power_psigma * power_psigma
            power_kurt *= float(nsim+1)*float(nsim)/(float(nsim-1)*float(nsim-2)*float(nsim-3)*power_psigma*power_psigma*power_psigma*power_psigma)
            power_kurt -= 3.*float(nsim-1)*float(nsim-1)/(float(nsim-2)*float(nsim-3))
        else:
            power_skew /= float(nsim)
            power_skew /= power_psigma * power_psigma * power_psigma
            power_kurt /= float(nsim)
            power_kurt /= power_psigma * power_psigma * power_psigma * power_psigma
            power_kurt -= 3.
    
        f=open(fname,"w")
        for i in range(0,power_k.size):
            f.write(str("%-.12e"%power_k[i])+" "+str("%-.12e"%power_skew[i])+" "+str("%-.12e"%power_kurt[i])+"\n")
        f.close()
            
    return power_k, power_skew, power_kurt
# ---------------------------------------------------------------------------- #



# ----------------------------- MEAN MASSFUNCTION ---------------------------- #
def mean_massfunction(mainpath = "", simset = "", isimmin = 1, isimmax = 2, noutput = 1, okprint = False):
    nsim = 0
    mf_binmin, mf_binmax, dummy = read_massfunction(file_path("massfunction", mainpath, simset, isimmin, noutput))
    mf_bin = (mf_binmin+mf_binmax)/2
    mf_mean = np.zeros(mf_bin.size)
    mf_sigma = np.zeros(mf_bin.size)
    for isim in range(isimmin, isimmax):
        current_file = file_path("massfunction", mainpath, simset, isim, noutput)
        if (okprint) : 
            print current_file
        dummy, dummy, mf_count = read_massfunction(current_file)
        mf_mean += mf_count
        nsim += 1
    mf_mean /= float(nsim)
    if (nsim > 1):
        for isim in range(isimmin, isimmax):
            current_file = file_path("massfunction", mainpath, simset, isim, noutput)
            if (okprint) : 
                print current_file
            dummy, dummy, mf_count = read_massfunction(current_file)
            mf_sigma += (mf_count-mf_mean)*(mf_count-mf_mean)
        mf_sigma /= float(nsim-1)
        mf_sigma = np.sqrt(mf_sigma)
    return mf_binmin, mf_binmax, mf_bin, mf_mean, mf_sigma
# ---------------------------------------------------------------------------- #



# -------------------------------- BACKUP MEAN ------------------------------- #
def backup_mean(mainpath = "", simset = "", growth_a = np.zeros(0), growth_dplus = np.zeros(0), fltformat = "%-.12e", outdir = "power", outprefix = "power_"):
    if (simset == "4096_furphase" or simset == "4096_otherphase"):
        isimmax = 1
        noutputmax = 31
    elif (simset == "4096_furphase_512"):
        isimmax = 512
        noutputmax = 9
    else:
        isimmax = 4096
        noutputmax = 9
    for ioutput in range(1, noutputmax+1):
        power_k, power_pmean, power_psigma = mean_power("power", mainpath, simset, 1, isimmax+1, ioutput, growth_a, growth_dplus, True)
        rpower_k, rpower_pmean, rpower_psigma = mean_power("renormalized", mainpath, simset, 1, isimmax+1, ioutput, growth_a, growth_dplus, True)
        f = open(outdir+"/"+outprefix+simset+"_"+str("%05d"%ioutput)+".txt", "w")
        f.write("# DEUS-PUR: Mean power spectrum - V. Reverdy 2013 - "+simset+" - [power_k, power_pmean, power_psigma, renormalized_pmean, renormalized_psigma]\n")
        for i in range(0, power_k.size):
            f.write(str(fltformat%power_k[i])+" "+str(fltformat%power_pmean[i])+" "+str(fltformat%power_psigma[i])+" "+str(fltformat%rpower_pmean[i])+" "+str(fltformat%rpower_psigma[i])+"\n")
        f.close()
    return
# ---------------------------------------------------------------------------- #



# ---------------------------------------------------------------------------- #
def mass_corrected_power(mainpath = "", simset = "", nsim = 1, noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), corr_type = "var_pres_smooth"):
    
    power_k, power_p = nyquist_power(mainpath, simset, nsim, noutput, aexp, growth_a, growth_dplus)
    
    if (corr_type == "var_pres" or corr_type == "var_pres_smooth" or corr_type == "var_pres_pl"):
        correction_smooth = correction_power(mainpath,noutput,aexp,growth_a,growth_dplus,corr_type)
        power_k, power_pmean, power_psigma = mean_power_all("nyquist",mainpath,noutput,aexp,growth_a,growth_dplus)
        power_k1024, power_pmean1024, power_psigma1024 = mean_power_all_1024("nyquist",mainpath,noutput,aexp,growth_a,growth_dplus)
        index = (power_k1024 <= power_k[power_k.size-1])
        power_pmean_1024=power_pmean1024[index]

        corrected_p = correction_smooth * power_p + power_pmean_1024 - correction_smooth * power_pmean
    else:
        if (simset == "4096_adaphase_256" or simset == "4096_otherphase_256" or simset == "4096_furphase_256"):
            correction_smooth = correction_power(mainpath,noutput,aexp,growth_a,growth_dplus,corr_type)
            #correction_smooth = np.poly1d(param)
            corrected_p = power_p / correction_smooth#(power_k)
            #corrected_p = power_p / correction
        else:
            print "No mass resolution correction implemented"
            corrected_p = power_p

    return power_k, corrected_p
# ---------------------------------------------------------------------------- #



# ---------------------------------------------------------------------------- #
def correction_power(mainpath = "", noutput = 1, aexp = 1.,growth_a = np.zeros(0),growth_dplus = np.zeros(0), corr_type = "var_pres_pl"):
    fname = "correction_"+corr_type+"_"+str("%05d"%noutput)+".txt"
    if(os.path.isfile(fname)):
        correction = np.loadtxt(fname,unpack=True)
    else:
        if (corr_type == "var_pres"):

            power_k, power_pmean, power_psigma = mean_power_all("nyquist",mainpath,noutput,aexp,growth_a,growth_dplus)
            power_k_CAMB, power_p_CAMB, dummy, dummy, dummy, dummy, dummy = read_data(mainpath)
            aexp_end = 1.
            dplus_a = extrapolate([aexp], growth_a, growth_dplus)
            dplus_end = extrapolate([aexp_end], growth_a, growth_dplus)
            plin = power_p_CAMB * dplus_a * dplus_a / (dplus_end * dplus_end)

            plin_interp = np.interp(power_k, power_k_CAMB, plin)

            N_k = np.zeros(power_k.size)
            delta_k = 0.0048
            for j in range(0, power_k.size):
                N_k[j]=656.25*656.25*656.25*power_k[j]*power_k[j]*delta_k/(2.*math.pi*math.pi)
            power_k1024, power_pmean1024, power_psigma1024 = mean_power_all_1024("nyquist",mainpath,noutput,aexp,growth_a,growth_dplus)
            index = (power_k1024 <= power_k[power_k.size-1])
            power_psigma_1024=power_psigma1024[index]

            pvar_norm = power_psigma_1024 * power_psigma_1024 * N_k / (2. * plin_interp * plin_interp)
            fit_par = np.polyfit(power_k,pvar_norm,9)
            fit = np.polyval(fit_par,power_k)

            #pvar_norm256 = power_psigma * power_psigma * N_k / (2. * plin_interp * plin_interp)
            #fit_par256 = np.polyfit(power_k,pvar_norm256,9)
            #fit256 = np.polyval(fit_par256,power_k)

            #correction = np.sqrt(fit) / np.sqrt(fit256) # 
            correction = np.sqrt(fit* (2.*plin_interp*plin_interp) / N_k) / power_psigma
            
            f=open(fname,"w")
            for i in range(0,correction.size):
                f.write(str("%-.12e"%correction[i])+"\n")
            f.close()

        elif (corr_type == "var_pres_smooth"):
    
            power_k, power_pmean, power_psigma = mean_power_all("nyquist",mainpath,noutput,aexp,growth_a,growth_dplus)

            power_k1024, power_pmean1024, power_psigma1024 = mean_power_all_1024("nyquist",mainpath,noutput,aexp,growth_a,growth_dplus)
            index = (power_k1024 <= power_k[power_k.size-1])
            power_psigma_1024=power_psigma1024[index]

            grade = 4
            fit_par = np.polyfit(power_k,power_psigma_1024/power_psigma,grade)
            correction = np.polyval(fit_par, power_k)
#            print fit_par

            params = Parameters()
            params.add('p0',value=1.,vary=False)
            par = np.zeros(fit_par.size)
            par[0] = 1.
            for i in range(1,fit_par.size):
                params.add('p'+str("%01d"%(i)),value=fit_par[grade-i],vary=True)
                par[i] = fit_par[grade-i]
            poly = lambda p,x: params['p0'].value + params['p1'].value * x + params['p2'].value * x * x + params['p3'].value * x * x * x + params['p4'].value * x * x * x * x
            #print params
#            print par

#            poly = np.poly1d(par)
            polyerr = lambda p,x,y: poly(p,x) - y

            fit = minimize(polyerr, params, args=(power_k,power_psigma_1024/power_psigma))
            for i in range(0,grade):
                par[i]=params['p'+str("%01d"%i)].value
                
            #print params
            print par

            #correction = np.polyval(par,power_k)
            x = power_k
            correction = params['p0'].value + params['p1'].value * x + params['p2'].value * x * x + params['p3'].value * x * x * x + params['p4'].value * x * x * x * x

            f=open(fname,"w")
            for i in range(0,correction.size):
                f.write(str("%-.12e"%correction[i])+"\n")
            f.close()

        elif (corr_type == "var_pres_pl"):
        
            power_k, power_pmean, power_psigma = mean_power_all("nyquist",mainpath,noutput,aexp,growth_a,growth_dplus)

            power_k1024, power_pmean1024, power_psigma1024 = mean_power_all_1024("nyquist",mainpath,noutput,aexp,growth_a,growth_dplus)
            index = (power_k1024 <= power_k[power_k.size-1])
            power_psigma_1024=power_psigma1024[index]

            params = Parameters()
            #params.add('p0',value=1.,vary=False)
            params.add('alpha',value=1.,vary=True)
            params.add('beta',value=2.,vary=True)

            powlaw = lambda p,x: 1.+params['alpha'].value * x ** (params['beta'].value)
            
            powerr = lambda p,x,y: powlaw(p,x) - y

            fit = minimize(powerr, params, args=(power_k,power_psigma_1024/power_psigma))
            
            x = power_k
            correction =1.+ params['alpha'].value * x **params['beta'].value

            print 'alpha', 'beta'
            print params['alpha'].value, params['beta'].value
            print 'err_alpha', 'err_beta'
            print params['alpha'].stderr, params['beta'].stderr
            f=open(fname,"w")
            for i in range(0,correction.size):
                f.write(str("%-.12e"%correction[i])+"\n")
            f.close()

        elif (corr_type == "mode"):
            power_k, power_p = nyquist_power(mainpath, "4096_furphase_256", 1, noutput, aexp, growth_a, growth_dplus)
            if (noutput == 9):
                correction = np.loadtxt("correction_z0.dat",unpack=True)
            elif (noutput == 4):
                correction = np.loadtxt("correction_z1.dat",unpack=True)
#    elif (noutput == 2):
            else:
                median_1024 = np.zeros(power_k.size)
                median_256 = np.zeros(power_k.size)
                for ik in range(0, power_k.size):
                    pp,npp = distrib_power("corrected", mainpath, "all_256", 1, 12289, noutput, 50, power_k[ik], aexp, growth_a, growth_dplus)
                    pp_1024, np_1024 = distrib_power("corrected", mainpath, "all_1024", 1, 89, noutput, 50, power_k[ik], aexp, growth_a, growth_dplus)
                    index = (np_1024 == max(np_1024) )
                    median = pp_1024[index]
                    median_1024[ik] = median[0]
                    index1 = (npp == max(npp))
                    median1 = pp[index1]
                    median_256[ik] = median1[0]
                    print median_256[ik]/median_1024[ik]
                correction_to_fit = median_256/median_1024

            fit_par = np.polyfit(power_k,correction_to_fit,13)
            correction = np.polyval(fit_par,power_k)
        else:
            n256 = random.randint(1,4096)
            nset = random.randint(1,3)
            #a = np.arange(1,88)
            #n1024 = random.shuffle(a)
            n1024 = random.randint(1,96)
#           print n256, nset, n1024
            if (nset==1):
                power_k, power_p = nyquist_power(mainpath, "4096_furphase_256", n256, noutput, aexp, growth_a, growth_dplus)
            elif (nset==2):
                power_k, power_p = nyquist_power(mainpath, "4096_otherphase_256", n256, noutput, aexp, growth_a, growth_dplus)
            elif (nset==3):
                power_k, power_p = nyquist_power(mainpath, "4096_adaphase_256", n256, noutput, aexp, growth_a, growth_dplus)
            if (n1024<=64):
                power_k_1024, power_p_1024 = nyquist_power(mainpath, "64_adaphase_1024", n1024, noutput, aexp, growth_a, growth_dplus)
            else:
                n1024=n1024-64
                power_k_1024, power_p_1024 = nyquist_power(mainpath, "64_curiephase_1024", n1024, noutput, aexp, growth_a, growth_dplus)
                index = ( power_k_1024 <= power_k[power_k.size-1])
                power_p1024 = power_p_1024[index]
            correction_to_fit = power_p/power_p1024
            fit_par = np.polyfit(power_k,correction_to_fit,2)
            correction = np.polyval(fit_par,power_k)
                
    return correction
# ---------------------------------------------------------------------------- #



# ---------------------------------------------------------------------------- #
def power_spectrum(powertype = "power", mainpath = "", simset = "", nsim = 1, noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), okprint = False):
    if (simset=="all_256"):
        if (okprint):
            print "setting simset to 4096_furphase_256 in function power_spectrum"
        simset="4096_furphase_256"
    if (powertype=="power"):
        power_k, power_p, dummy = read_power(file_path("power", mainpath, simset, nsim, noutput))
    elif (powertype=="nyquist"):
        power_k, power_p = nyquist_power(mainpath, simset, nsim, noutput, aexp, growth_a, growth_dplus)
    elif (powertype=="renormalized"):
        power_k, power_p = renormalized_power(mainpath, simset, nsim, noutput, growth_a, growth_dplus)
    elif (powertype=="corrected"):
        power_k, power_p = corrected_power(mainpath, simset, aexp, nsim, noutput, growth_a, growth_dplus)
    elif (powertype=="mcorrected"):
        power_k, power_p = mass_corrected_power(mainpath, simset, nsim, noutput, aexp, growth_a, growth_dplus)
    else:
        if (okprint):
            print "powetype not existent in function power_spectrum"

    return power_k, power_p
# ---------------------------------------------------------------------------- #
