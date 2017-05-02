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



# -------------------------------- MEAN POWER (ALTERNATIVE) -------------------------------- #
def mean_power(powertype = "power", mainpath = "", simset = "", isimmin = 1, isimmax = 2, noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), nmodel = 0, okprint = False, store = False):
    
    nsim = isimmax-isimmin

    fname = output_file_name("mean",powertype,simset,isimmin,isimmax,noutput,nmodel)
    if(os.path.isfile(fname) and not store):
        if (okprint):
            print "Reading mean spectrum from file: ", fname
        power_k, power_pmean, power_psigma = np.loadtxt(fname,unpack=True)
    else:
        if (okprint):
            print "Computing mean and standard deviation of spectra"
        if (powertype=="linear"):
            power_k, power_pmean = power_spectrum("linear",mainpath,simset.name,1,noutput,aexp,growth_a,growth_dplus,nmodel,okprint)
            power_psigma = np.sqrt(2./simset.N_k(power_k))*power_pmean
        else:
            true_simset, true_isimmin = sim_iterator(simset.name, isimmin)
            power_k, dummy = power_spectrum(powertype,mainpath,true_simset,true_isimmin,noutput,aexp,growth_a,growth_dplus,nmodel)
            power_p = np.zeros((nsim,power_k.size))
            for isim in xrange(isimmin, isimmax):
                isim0=isim-isimmin
                true_simset,true_isim = sim_iterator(simset.name, isim)
                if (okprint):
                    print true_simset, true_isim
                dummy, power_p[isim0] = power_spectrum(powertype,mainpath,true_simset,true_isim,noutput,aexp,growth_a,growth_dplus,nmodel,okprint)

            power_pmean = np.mean(power_p,axis=0)
            if (nsim > 1):
                power_psigma = np.std(power_p,axis=0,ddof=1)
            else:
                if (okprint):
                    print "Standard deviation not computed because there is only one simulation"

        if (store):
            true_simset,true_isim = sim_iterator(simset.name, 1)
            if np.isclose(aexp,read_info(input_file_name("info",mainpath,true_simset,true_isim,noutput)),atol=1.e-2):
                if (okprint):
                    print "Writing file: ",fname
                f = open(fname, "w")
                for i in xrange(0, power_k.size):
                    f.write(str("%-.12e"%power_k[i])+" "+str("%-.12e"%power_pmean[i])+" "+str("%-.12e"%power_psigma[i])+"\n")
                f.close()
            else:
                print "Not writing file ",fname," because aexp ", aexp," does not correspond to noutput", noutput
                print "input aexp: ", aexp,", aexp from sim: ",read_info(input_file_name("info",mainpath,true_simset,true_isim,noutput))

    return power_k, power_pmean, power_psigma
# ---------------------------------------------------------------------------- #



# --------------------- PDF OF POWER SPECTRA --------------------------- #
def distrib_power(powertype = "power", mainpath = "", simset = "", isimmin = 1, isimmax = 2, noutput = 1, nbin = 50, kref = 0.2, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), nmodel = 0, okprint = False, store = False):
    
    nsim = isimmax - isimmin
    power_values = np.zeros(nsim)

    fname = output_file_name("distrib_k"+str(kref),powertype,simset,isimmin,isimmax,noutput,nmodel)
    if (os.path.isfile(fname) and not store):
        bincenter, npower_bin = np.loadtxt(fname,unpack=True)
    else:
        for isim in xrange(1, nsim+1):
            true_simset, true_isim = sim_iterator(simset.name,isim)
            if (okprint) :
                print true_simset, true_isim
            power_k, power_p = power_spectrum(powertype,mainpath,true_simset,true_isim,noutput,aexp,growth_a,growth_dplus,nmodel)
            if (kref>power_k[power_k.size-1] or kref<power_k[0]):
                print "WARNING: reference k value outside simulation k range in distrib_power"

            power_values[isim-1]=power_p[np.searchsorted(power_k,kref)]
        
        npower_bin, bins = np.histogram(power_values,nbin)
        npower_bin = np.asfarray(npower_bin)/float(nsim)
        bincenter = 0.5*(bins[1:]+bins[:-1])
        
        if (store):
            true_simset,true_isim = sim_iterator(simset.name, 1)
            if np.isclose(aexp,read_info(input_file_name("info",mainpath,true_simset,true_isim,noutput)),atol=1.e-2):
                if (okprint):
                    print "Writing file: ",fname
                f = open(fname, "w")
                for i in xrange(0, bincenter.size):
                    f.write(str("%-.12e"%bincenter[i])+" "+str("%-.12e"%npower_bin[i])+"\n")
                f.close()
            else:
                print "Not writing file ",fname," because aexp ", aexp," does not correspond to noutput", noutput
                print "input aexp: ", aexp,", aexp from sim: ",read_info(input_file_name("info",mainpath,true_simset,true_isim,noutput))

    return bincenter, npower_bin
# ---------------------------------------------------------------------------- #



# ------------------ HIGH MOMENTS OF SPECTRA PDF ---------------------- #
def high_moments(powertype = "power", mainpath = "", simset = "", isimmin = 1, isimmax = 2, noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), nmodel = 0, unbiased = True, okprint = False, store = True):

    nsim = isimmax-isimmin
    
    if(unbiased):
        bias="unbiased"
    else:
        bias="biased"


    fname = output_file_name("high_moments_"+bias,powertype,simset,isimmin,isimmax,noutput,nmodel)
    if(os.path.isfile(fname) and not store):
        power_k, power_skew, power_kurt = np.loadtxt(fname,unpack=True)
    else:
        power_k, power_pmean, power_psigma = mean_power(powertype, mainpath, simset, isimmin, isimmax, noutput, aexp, growth_a, growth_dplus, nmodel, okprint, store)
        power_skew = np.zeros(power_k.size)
        power_kurt = np.zeros(power_k.size)
    
        for isim in xrange(isimmin, isimmax+1):
            true_simset, true_isim = sim_iterator(simset.name,isim)
            if (okprint) :
                print true_simset,true_isim
            power_k, power_p = power_spectrum(powertype,mainpath,true_simset,true_isim,noutput,aexp,growth_a,growth_dplus,nmodel)
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

        if (store):
            true_simset,true_isim = sim_iterator(simset.name, 1)
            if np.isclose(aexp,read_info(input_file_name("info",mainpath,true_simset,true_isim,noutput)),atol=1.e-2):
                if (okprint):
                    print "Writing file: ",fname
                f=open(fname,"w")
                for i in xrange(0,power_k.size):
                    f.write(str("%-.12e"%power_k[i])+" "+str("%-.12e"%power_skew[i])+" "+str("%-.12e"%power_kurt[i])+"\n")
                f.close()
            else:
                print "Not writing file ",fname," because aexp ", aexp," does not correspond to noutput", noutput
                print "input aexp: ", aexp,", aexp from sim: ",read_info(input_file_name("info",mainpath,true_simset,true_isim,noutput))
    
    return power_k, power_skew, power_kurt
# ---------------------------------------------------------------------------- #



# ------------------------- SPECTRUM CORRECTED FOR MASS RES EFFECT --------------------------- #
def mass_corrected_power(mainpath = "", simset = "", nsim = 1, noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), corr_type = "var_pres_smooth", okprint=False, store=False):
    
    if (simset.npart!=256):
        print "WARNING: using mass resolution correction outside its applicability"

    fname = input_file_name("power",mainpath,simset.name,nsim,noutput,0,okprint,"mcorrected")
    if (os.path.isfile(fname) and not store):
        power_k, corrected_p = np.loadtxt(fname,unpack=True)
    else:
        power_k, power_p = nyquist_power(mainpath,simset.name,nsim,noutput,aexp,growth_a,growth_dplus,okprint=okprint)
        correction_smooth = correction_power(mainpath,simset.name,noutput,aexp,growth_a,growth_dplus,corr_type,okprint=okprint,store=store)

        if (corr_type == "var_pres" or corr_type == "var_pres_smooth" or corr_type == "var_pres_pl"):
            simset_256 = DeusPurSet("all_256")
            power_k, power_pmean, power_psigma = mean_power("nyquist",mainpath,simset_256.name,1,simset_256.nsimmax+1,noutput,aexp,growth_a,growth_dplus,okprint=okprint,store=store)
            simset_1024 = DeusPurSet("all_1024")
            power_k1024, power_pmean1024, power_psigma1024 = mean_power("nyquist",mainpath,simset_1024.name,1,simset_1024.nsimmax+1,noutput,aexp,growth_a,growth_dplus,okprint=okprint,store=store)
            index = (power_k1024 <= power_k[power_k.size-1])
            power_pmean_1024=power_pmean1024[index]
            corrected_p = correction_smooth * power_p + power_pmean_1024 - correction_smooth * power_pmean
        else:
            corrected_p = power_p / correction_smooth

        if (store):
            true_simset,true_isim = sim_iterator(simset.name, 1)
            if np.isclose(aexp,read_info(input_file_name("info",mainpath,true_simset,true_isim,noutput)),atol=1.e-2):
                if (okprint):
                    print "Writing file: ",fname
                f=open(fname,"w")
                for i in xrange(0,power_k.size):
                    f.write(str("%-.12e"%power_k[i])+" "+str("%-.12e"%corrected_p[i])+"\n")
                f.close()
            else:
                print "Not writing file ",fname," because aexp ", aexp," does not correspond to noutput", noutput
                print "input aexp: ", aexp,", aexp from sim: ",read_info(input_file_name("info",mainpath,true_simset,true_isim,noutput))

    return power_k, corrected_p
# ---------------------------------------------------------------------------- #



# ------------------------- CORRECTION TO THE LOW RES SPECTRA ------------------------------ #
def correction_power(mainpath = "", simset="",
                     noutput = 1, aexp = 1.,growth_a = np.zeros(0),growth_dplus = np.zeros(0), corr_type = "var_pres_smooth",okprint = False, store = False):
    
    fname = "correction_"+corr_type+"_"+str("%05d"%noutput)+".txt"
    if(os.path.isfile(fname) and not store):
        correction = np.loadtxt(fname,unpack=True)
    else:
        simset_256 = DeusPurSet("all_256")
        simset_1024 = DeusPurSet("all_1024")

        if (corr_type == "var_pres"):
            if (okprint):
                print "Mass resolution correction for mean and variance of the power spectrum with no smoothing."
            power_k, power_pmean, power_psigma = mean_power("nyquist",mainpath,simset_256.name,1,simset_256.nsimmax+1,noutput,aexp,growth_a,growth_dplus,okprint)
            power_k_CAMB, power_p_CAMB = read_power_camb(mainpath)
            aexp_end = 1.
            dplus_a = extrapolate([aexp], growth_a, growth_dplus)
            dplus_end = extrapolate([aexp_end], growth_a, growth_dplus)
            plin = power_p_CAMB * dplus_a * dplus_a / (dplus_end * dplus_end)
            plin_interp = np.interp(power_k, power_k_CAMB, plin)
            N_k = simset.N_k(power_k)
            
            power_k1024, power_pmean1024, power_psigma1024 = mean_power("nyquist",mainpath,simset_1024.name,1,simset_1024.nsimmax+1,noutput,aexp,growth_a,growth_dplus,okprint)
            index = (power_k1024 <= power_k[power_k.size-1])
            power_psigma_1024=power_psigma1024[index]

            pvar_norm = power_psigma_1024 * power_psigma_1024 * N_k / (2. * plin_interp * plin_interp)
            fit_par = np.polyfit(power_k,pvar_norm,9)
            fit = np.polyval(fit_par,power_k)
            
            correction = np.sqrt(fit* (2.*plin_interp*plin_interp) / N_k) / power_psigma

        elif (corr_type == "var_pres_smooth"):
            if (okprint):
                print "Mass resolution correction for mean and variance of the power spectrum with polynomial smoothing (4th order)."
            power_k, power_pmean, power_psigma = mean_power("nyquist",mainpath,simset_256.name,1,simset_256.nsimmax+1,noutput,aexp,growth_a,growth_dplus,okprint)
            power_k1024, power_pmean1024, power_psigma1024 = mean_power("nyquist",mainpath,simset_1024.name,1,simset_1024.nsimmax+1,noutput,aexp,growth_a,growth_dplus,okprint)
            index = (power_k1024 <= power_k[power_k.size-1])
            power_psigma_1024=power_psigma1024[index]

            grade = 4
            fit_par = np.polyfit(power_k,power_psigma_1024/power_psigma,grade)
            correction = np.polyval(fit_par, power_k)

            params = Parameters()
            params.add('p0',value=1.,vary=False)
            par = np.zeros(fit_par.size)
            par[0] = 1.
            for i in xrange(1,fit_par.size):
                params.add('p'+str("%01d"%(i)),value=fit_par[grade-i],vary=True)
                par[i] = fit_par[grade-i]
            poly = lambda p,x: params['p0'].value + params['p1'].value * x + params['p2'].value * x * x + params['p3'].value * x * x * x + params['p4'].value * x * x * x * x

            polyerr = lambda p,x,y: poly(p,x) - y

            fit = minimize(polyerr, params, args=(power_k,power_psigma_1024/power_psigma))
            for i in xrange(0,grade):
                par[i]=params['p'+str("%01d"%i)].value

            if (okprint):
                print par

            x = power_k
            correction = params['p0'].value + params['p1'].value * x + params['p2'].value * x * x + params['p3'].value * x * x * x + params['p4'].value * x * x * x * x

        elif (corr_type == "var_pres_pl"):
            if (okprint):
                print "Mass resolution correction for mean and variance of the power spectrum with power-law smoothing"
            power_k, power_pmean, power_psigma = mean_power("nyquist",mainpath,simset_256.name,1,simset_256.nsimmax+1,noutput,aexp,growth_a,growth_dplus,okprint)
            power_k1024, power_pmean1024, power_psigma1024 = mean_power("nyquist",mainpath,simset_1024.name,1,simset_1024.nsimmax+1,noutput,aexp,growth_a,growth_dplus,okprint)
            index = (power_k1024 <= power_k[power_k.size-1])
            power_psigma_1024=power_psigma1024[index]

            params = Parameters()
            params.add('alpha',value=1.,vary=True)
            params.add('beta',value=2.,vary=True)

            powlaw = lambda p,x: 1.+params['alpha'].value * x ** (params['beta'].value)
            
            powerr = lambda p,x,y: powlaw(p,x) - y

            fit = minimize(powerr, params, args=(power_k,power_psigma_1024/power_psigma))
            
            x = power_k
            correction =1.+ params['alpha'].value * x **params['beta'].value

            if (okprint):
                print 'alpha', 'beta'
                print params['alpha'].value, params['beta'].value
                print 'err_alpha', 'err_beta'
                print params['alpha'].stderr, params['beta'].stderr

        elif (corr_type=="mean"):
            if (okprint):
                print "Mass resolution correction for mean power spectrum"
            power_k, power_pmean, power_psigma = mean_power("nyquist",mainpath,simset_256.name,1,simset_256.nsimmax+1,noutput,aexp,growth_a,growth_dplus,okprint)
            power_k1024, power_pmean1024, power_psigma1024 = mean_power("nyquist",mainpath,simset_1024.name,1,simset_1024.nsimmax+1,noutput,aexp,growth_a,growth_dplus,okprint)
            index = (power_k1024 <= power_k[power_k.size-1])
            power_pmean_1024=power_pmean1024[index]
            correction = power_pmean/power_pmean1024

        else:
            print "WARNING: unknown corr_type in correction_power"

        if (store):
            if (okprint):
                print "Writing file: ",fname
            f=open(fname,"w")
            for i in xrange(0,correction.size):
                f.write(str("%-.12e"%correction[i])+"\n")
            f.close()
    
    return correction
# ---------------------------------------------------------------------------- #



# ----------------------- WRAPPER FOR ALL POWER TYPES ------------------------- #
def power_spectrum(powertype = "power", mainpath = "", simset = "", nsim = 1, noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), nmodel = 0, okprint = False, store = False):

    setname, nsim = sim_iterator(simset, nsim)

    if (setname=="Minerva" or setname=="COLA"):
        power_k, dummy, power_p, dummy, dummy, dummy = np.genfromtxt(input_file_name("power", mainpath, simset, nsim, noutput, nmodel, okprint, powertype))
    else:
        if (powertype=="power"):
            power_k, power_p, dummy = read_power(input_file_name("power", mainpath, setname, nsim, noutput, nmodel))
        elif (powertype=="nyquist"):
            power_k, power_p = nyquist_power(mainpath, setname, nsim, noutput, aexp, growth_a, growth_dplus, nmodel, okprint, store)
        elif (powertype=="renormalized"):
            power_k, power_p = renormalized_power(mainpath, setname, nsim, noutput, growth_a, growth_dplus, nmodel, okprint, store)
        elif (powertype=="corrected"):
            power_k, power_p = corrected_power(mainpath, setname, aexp, nsim, noutput, growth_a, growth_dplus, nmodel, okprint, store)
        elif (powertype=="mcorrected"):
            power_k, power_p = mass_corrected_power(mainpath, setname, nsim, noutput, aexp, growth_a, growth_dplus, okprint, store)
        elif (powertype=="linear"):
            if (nmodel==0):
                model="lcdmw7"
            else:
                model="model"+str(int(nmodel)).zfill(5)
            power_k_CAMB, power_p_CAMB = read_power_camb(mainpath, model)
            power_k_nocut, dummy, dummy = read_power(input_file_name("power", mainpath, setname, nsim, noutput, nmodel))
            aexp_end = 1.
            dplus_a = extrapolate([aexp], growth_a, growth_dplus)
            dplus_end = extrapolate([aexp_end], growth_a, growth_dplus)
            plin = power_p_CAMB * dplus_a * dplus_a / (dplus_end * dplus_end)
            idx = (power_k_nocut < simset.nyquist)
            power_k = power_k_nocut[idx]
            power_p = np.interp(power_k, power_k_CAMB, plin)
            if (store):
                if (okprint):
                    print "Writing file: ",fname
                f=open(fname,"w")
                for i in xrange(0,power_k.size):
                    f.write(str("%-.12e"%power_k[i])+" "+str("%-.12e"%corrected_p[i])+"\n")
                f.close()


        elif (powertype=="linear_mock"):
            if (nmodel==0):
                model="lcdmw7"
            else:
                model="model"+str(int(nmodel)).zfill(5)
            power_k_CAMB, power_p_CAMB = read_power_camb(mainpath, model)
            power_k, dummy, dummy = read_power(input_file_name("power", mainpath, setname, nsim, noutput, nmodel))
            aexp_end = 1.
            dplus_a = extrapolate([aexp], growth_a, growth_dplus)
            dplus_end = extrapolate([aexp_end], growth_a, growth_dplus)
            plin = power_p_CAMB * dplus_a * dplus_a / (dplus_end * dplus_end)
                                                         
            plin_interp = np.interp(power_k, power_k_CAMB, plin)
            N_k = simset.N_k(power_k)
            noise = np.sqrt(2./N_k)*(plin_interp)
            error = np.random.normal(0.,noise,noise.size)
            power_p = plin_interp + error
        else:
            print "powertype not existent in function power_spectrum"
            sys.exit()

    return power_k, power_p
# ---------------------------------------------------------------------------- #
