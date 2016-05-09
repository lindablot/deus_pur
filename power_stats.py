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
    
    if (type(simset) is str):
        simset=DeusPurSet(simset)
    nsim = isimmax-isimmin

    fname = "mean_"+powertype+"_"+str("%05d"%noutput)+"_"
    if (nsim==simset.nsimmax):
        if(simset.cosmo):
            fname = fname+"cosmo_model"+str(int(nmodel))+".txt"
        else:
            fname = fname+simset.name+".txt"
    else:
        if(simset.cosmo):
            fname = fname+"cosmo_model"+str(int(nmodel))+"_"+str(isimmin)+"_"+str(isimmax)+".txt"
        else:
            fname = fname+simset.name+"_"+str(isimmin)+"_"+str(isimmax)+".txt"

    if(os.path.isfile(fname)):
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
                dummy, power_p[isim0] = power_spectrum(powertype,mainpath,true_simset,true_isim,noutput,aexp,growth_a,growth_dplus,nmodel,okprint=okprint)

            power_pmean = np.mean(power_p,axis=0)
            if (nsim > 1):
                power_psigma = np.std(power_p,axis=0,ddof=1)
            else:
                if (okprint):
                    print "Standard deviation not computed because there is only one simulation"

        if (store):
            f = open(fname, "w")
            for i in xrange(0, power_k.size):
                f.write(str("%-.12e"%power_k[i])+" "+str("%-.12e"%power_pmean[i])+" "+str("%-.12e"%power_psigma[i])+"\n")
            f.close()

    return power_k, power_pmean, power_psigma
# ---------------------------------------------------------------------------- #



# ------------------------------- MEAN POWER ON ALL 512 REBINNED ------------------------ #
def mean_power_all_512r256(powertype = "power", mainpath = "", noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), okprint = False):
    
    def N_k(power_k=np.zeros(0),L_box=656.25):
        return math.pi/24.+power_k*power_k*L_box*L_box/(2.*math.pi)
    
    fname = "mean_"+powertype+"_512r256_"+str("%05d"%noutput)+".txt"
    simset = "4096_furphase_512"
    if(os.path.isfile(fname)):
        power_k256, power_pmean, power_psigma = np.loadtxt(fname,unpack=True)
    else:
        nsim = 512
        power_k256, dummy = power_spectrum(powertype,mainpath,"all_256",1,noutput,aexp,growth_a,growth_dplus)
        power_pmean = np.zeros(power_k256.size)
        power_psigma = np.zeros(power_k256.size)
        
        for isim in xrange(1,nsim+1):
            if (okprint) :
                current_file = file_path("power", mainpath, simset, isim, noutput)
                print current_file
            power_k, power_p = power_spectrum(powertype,mainpath,simset,isim,noutput,aexp,growth_a,growth_dplus,okprint)
            nh = N_k(power_k,1312.5)
            pr256 = np.zeros(power_k256.size)
            for ik in xrange(0,power_k256.size-1):
                pr256[ik]=1./(nh[2*ik+1]/2.+nh[2*ik+2]+nh[2*ik+3]/2.)*(nh[2*ik+1]/2.*power_p[2*ik+1]+nh[2*ik+2]*power_p[2*ik+2]+nh[2*ik+3]/2.*power_p[2*ik+3])
            pr256[power_k256.size-1]=1./(nh[power_p.size-2]+nh[power_p.size-1])*(nh[power_p.size-1]*power_p[power_p.size-1]+nh[power_p.size-2]*power_p[power_p.size-2])
            power_pmean += pr256

        power_pmean /= float(nsim)

        for isim in xrange(1,nsim+1):
            if (okprint) :
                current_file = file_path("power", mainpath, simset, isim, noutput)
                print current_file
            dummy, power_p = power_spectrum(powertype,mainpath,simset,isim,noutput,aexp,growth_a,growth_dplus,okprint)
            for ik in xrange(0,power_k256.size-1):
                pr256[ik]=1./(nh[2*ik+1]/2.+nh[2*ik+2]+nh[2*ik+3]/2.)*(nh[2*ik+1]/2.*power_p[2*ik+1]+nh[2*ik+2]*power_p[2*ik+2]+nh[2*ik+3]/2.*power_p[2*ik+3])
            pr256[power_k256.size-1]=1./(nh[power_p.size-2]+nh[power_p.size-1])*(nh[power_p.size-1]*power_p[power_p.size-1]+nh[power_p.size-2]*power_p[power_p.size-2])
            power_psigma += (pr256-power_pmean)*(pr256-power_pmean)
        power_psigma /= float(nsim-1)
        power_psigma = np.sqrt(power_psigma)
        
        f = open(fname, "w")
        for i in xrange(0, power_k256.size):
            f.write(str("%-.12e"%power_k256[i])+" "+str("%-.12e"%power_pmean[i])+" "+str("%-.12e"%power_psigma[i])+"\n")
        f.close()

    return power_k256, power_pmean, power_psigma
# ---------------------------------------------------------------------------- #



# --------------------- PDF OF POWER SPECTRA --------------------------- #
def distrib_power(powertype = "power", mainpath = "", simset = "", isimmin = 1, isimmax = 2, noutput = 1, nbin = 50, kref = 0.2, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), nmodel = 0, okprint = False, store = False):
    
    if (okprint):
        print "Entering distrib_power"
    if (type(simset) is str):
        simset=DeusPurSet(simset)

    nsim = isimmax - isimmin
    power_values = np.zeros(nsim)

    fname = "distrib_"+powertype+"_"+str("%05d"%noutput)+"_k"+str(kref)+"_"
    if (nsim==simset.nsimmax):
        if(simset.cosmo):
            fname = fname+"cosmo_model"+str(int(nmodel))+".txt"
        else:
            fname = fname+simset.name+".txt"
    else:
        if(simset.cosmo):
            fname = fname+"cosmo_model"+str(int(nmodel))+"_"+str(isimmin)+"_"+str(isimmax)+".txt"
        else:
            fname = fname+simset.name+"_"+str(isimmin)+"_"+str(isimmax)+".txt"

    if (os.path.isfile(fname)):
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
            f = open(fname, "w")
            for i in xrange(0, bincenter.size):
                f.write(str("%-.12e"%bincenter[i])+" "+str("%-.12e"%npower_bin[i])+"\n")
            f.close()

    return bincenter, npower_bin
# ---------------------------------------------------------------------------- #



# ------------------ HIGH MOMENTS OF SPECTRA PDF ---------------------- #
def high_moments(powertype = "power", mainpath = "", simset = "", isimmin = 1, isimmax = 2, noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), nmodel = 0, unbiased = True, okprint = False, store = True):

    if (type(simset) is str):
        simset=DeusPurSet(simset)
    nsim = isimmax-isimmin
    
    if(unbiased):
        bias="unbiased"
    else:
        bias="biased"

    fname="high_moments_"+bias+"_"+powertype+"_"+simset.name+"_"+str("%05d"%noutput)+".txt"
    if(os.path.isfile(fname)):
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
            f=open(fname,"w")
            for i in xrange(0,power_k.size):
                f.write(str("%-.12e"%power_k[i])+" "+str("%-.12e"%power_skew[i])+" "+str("%-.12e"%power_kurt[i])+"\n")
            f.close()
            
    return power_k, power_skew, power_kurt
# ---------------------------------------------------------------------------- #



# ------------------------- SPECTRUM CORRECTED FOR MASS RES EFFECT --------------------------- #
def mass_corrected_power(mainpath = "", simset = "", nsim = 1, noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), corr_type = "var_pres_smooth", okprint=False, store=False):
    
    if (type(simset) is str):
        simset=DeusPurSet(simset)
    if (simset.npart!=256):
        print "WARNING: using mass resolution correction outside its applicability"

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
        fname = file_path("power",mainpath,simset.name,nsim,noutput,0,okprint,"mcorrected")
        if (okprint):
            print "Writing file: ",fname
        f=open(fname,"w")
        for i in xrange(0,power_k.size):
            f.write(str("%-.12e"%power_k[i])+" "+str("%-.12e"%corrected_p[i])+"\n")
        f.close()

    return power_k, corrected_p
# ---------------------------------------------------------------------------- #



# ------------------------- CORRECTION TO THE LOW RES SPECTRA ------------------------------ #
def correction_power(mainpath = "", simset="",
                     noutput = 1, aexp = 1.,growth_a = np.zeros(0),growth_dplus = np.zeros(0), corr_type = "var_pres_smooth",okprint = False, store = False):
    if (type(simset) is str):
        simset=DeusPurSet(simset)
    
    fname = "correction_"+corr_type+"_"+str("%05d"%noutput)+".txt"
    if(os.path.isfile(fname)):
        correction = np.loadtxt(fname,unpack=True)
    else:
        simset_256 = DeusPurSet("all_256")
        simset_1024 = DeusPurSet("all_1024")

        if (corr_type == "var_pres"):
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

            print 'alpha', 'beta'
            print params['alpha'].value, params['beta'].value
            print 'err_alpha', 'err_beta'
            print params['alpha'].stderr, params['beta'].stderr

        elif (corr_type == "mode"):
            print "WARNING: obsolete type of correction for mass resolution effect:", corr_type
            power_k, power_p = nyquist_power(mainpath, "4096_furphase_256", 1, noutput, aexp, growth_a, growth_dplus)
            median_1024 = np.zeros(power_k.size)
            median_256 = np.zeros(power_k.size)
            for ik in xrange(0, power_k.size):
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

        elif (corr_type == "random"):
            print "WARNING: obsolete type of correction for mass resolution effect:", corr_type
            n256 = random.randint(1,4096)
            nset = random.randint(1,3)
            n1024 = random.randint(1,96)
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

        elif (corr_type=="mean"):
            power_k, power_pmean, power_psigma = mean_power("nyquist",mainpath,simset_256.name,1,simset_256.nsimmax+1,noutput,aexp,growth_a,growth_dplus,okprint)
            power_k1024, power_pmean1024, power_psigma1024 = mean_power("nyquist",mainpath,simset_1024.name,1,simset_1024.nsimmax+1,noutput,aexp,growth_a,growth_dplus,okprint)
            index = (power_k1024 <= power_k[power_k.size-1])
            power_pmean_1024=power_pmean1024[index]
            correction = power_pmean/power_pmean1024

        else:
            print "WARNING: unknown corr_type in correction_power"

        if (store):
            f=open(fname,"w")
            for i in xrange(0,correction.size):
                f.write(str("%-.12e"%correction[i])+"\n")
            f.close()
    
    return correction
# ---------------------------------------------------------------------------- #



# ----------------------- WRAPPER FOR ALL POWER TYPES ------------------------- #
def power_spectrum(powertype = "power", mainpath = "", simset = "", nsim = 1, noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), nmodel = 0, okprint = False, store = False):
    if (type(simset) is str):
        simset=DeusPurSet(simset)
    if (simset.composite):
        simset.name, nsim = sim_iterator(simset.name, nsim)
    if (powertype=="power"):
        power_k, power_p, dummy = read_power(file_path("power", mainpath, simset.name, nsim, noutput, nmodel))
    elif (powertype=="nyquist"):
        power_k, power_p = nyquist_power(mainpath, simset.name, nsim, noutput, aexp, growth_a, growth_dplus, nmodel, okprint=okprint, store=store)
    elif (powertype=="renormalized"):
        power_k, power_p = renormalized_power(mainpath, simset.name, nsim, noutput, growth_a, growth_dplus, nmodel, okprint=okprint, store=store)
    elif (powertype=="corrected"):
        power_k, power_p = corrected_power(mainpath, simset.name, aexp, nsim, noutput, growth_a, growth_dplus, nmodel, okprint=okprint, store=store)
    elif (powertype=="mcorrected"):
        power_k, power_p = mass_corrected_power(mainpath, simset.name, nsim, noutput, aexp, growth_a, growth_dplus, okprint=okprint, store=store)
    elif (powertype=="linear"):
        if (nmodel==0):
            model="lcdmw7"
        else:
            model="model"+str(int(nmodel)).zfill(5)
        power_k_CAMB, power_p_CAMB = read_power_camb(mainpath, model)
        power_k_nocut, dummy, dummy = read_power(file_path("power", mainpath, simset.name, nsim, noutput, nmodel))
        aexp_end = 1.
        dplus_a = extrapolate([aexp], growth_a, growth_dplus)
        dplus_end = extrapolate([aexp_end], growth_a, growth_dplus)
        plin = power_p_CAMB * dplus_a * dplus_a / (dplus_end * dplus_end)
        idx = (power_k_nocut < simset.nyquist)
        power_k = power_k_nocut[idx]
        power_p = np.interp(power_k, power_k_CAMB, plin)

    elif (powertype=="linear_mock"):
        if (nmodel==0):
            model="lcdmw7"
        else:
            model="model"+str(int(nmodel)).zfill(5)
        power_k_CAMB, power_p_CAMB = read_power_camb(mainpath, model)
        power_k, dummy, dummy = read_power(file_path("power", mainpath, simset.name, nsim, noutput, nmodel))
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
