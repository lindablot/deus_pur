#!/usr/bin/env python
# power_types.py - Vincent Reverdy (vince.rev@gmail.com) - 2013
# ---------------------------------- IMPORT ---------------------------------- #
import math
import glob
import os
import sys
import numpy as np
from read_files import *
import scipy.interpolate as itl
# ---------------------------------------------------------------------------- #


# ------------------------------- EXTRAPOLATE -------------------------------- #
def extrapolate(value_x, array_x, array_y):
    value_y = 0.
    if (value_x < array_x[0]):
        value_y = array_y[0]+(value_x-array_x[0])*(array_y[0]-array_y[1])/(array_x[0]-array_x[1])
    elif (value_x > array_x[-1]):
        value_y = array_y[-1]+(value_x-array_x[-1])*(array_y[-1]-array_y[-2])/(array_x[-1]-array_x[-2])
    else:
        value_y = np.interp(value_x, array_x, array_y)
    return value_y
# ---------------------------------------------------------------------------- #



# --------------------------- RENORMALIZED POWER ----------------------------- #
def renormalized_power(mainpath = "", simset = "", nsim = 1, noutput = 1, growth_a = np.zeros(0), growth_dplus = np.zeros(0)):
    power_k, power_p_ini, dummy = read_power(file_path("power", mainpath, simset, nsim, 1))
    power_k, power_p_end, dummy = read_power(file_path("power", mainpath, simset, nsim, noutput))
    aexp_ini = read_info(file_path("info", mainpath, simset, nsim, 1))
    aexp_end = read_info(file_path("info", mainpath, simset, nsim, noutput))
    dplus_ini = extrapolate([aexp_ini], growth_a, growth_dplus)
    dplus_end = extrapolate([aexp_end], growth_a, growth_dplus)
    power_p = (power_p_end*dplus_ini*dplus_ini)/(power_p_ini*dplus_end*dplus_end)
    return power_k, power_p
# ---------------------------------------------------------------------------- #



# ----------------------------- CORRECTED POWER ------------------------------ #
def corrected_power(mainpath = "", simset = "", aexp = 0., nsim = 1, noutput = 1, growth_a = np.zeros(0), growth_dplus = np.zeros(0)):
    power_k, power_p_raw, dummy = read_power(file_path("power", mainpath, simset, nsim, noutput))
    if (aexp != 0.):
        aexp_raw = read_info(file_path("info", mainpath, simset, nsim, noutput))
        dplus_raw = extrapolate([aexp_raw], growth_a, growth_dplus)
        dplus = extrapolate([aexp], growth_a, growth_dplus)
        power_p = (power_p_raw*dplus*dplus)/(dplus_raw*dplus_raw)
    else:
        power_p = power_p_raw
    return power_k, power_p
# ---------------------------------------------------------------------------- #



# ----------------------------- NYQUIST CUT POWER ------------------------------ #
def nyquist_power(mainpath = "", simset = "", nsim = 1, noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0)):
    pi = math.pi
    power_k, power_p = corrected_power(mainpath,simset,aexp,nsim,noutput,growth_a,growth_dplus)
    if (simset == "4096_furphase"):
        nyquist = (pi / 10000.) * 4096.#(4096. / 2.)
    elif (simset == "4096_otherphase"):
        nyquist= (pi / 10000.)* 4096.#(4096. / 2.)
    elif (simset == "4096_furphase_512"):
        nyquist= (pi / 1312.5)*512. #(512. / 2.)
    elif (simset == "4096_furphase_256"):
        nyquist= (pi / 656.25)*256.# (256. / 2.)
    elif (simset == "4096_otherphase_256"):
        nyquist= (pi / 656.25)*256.# (256. / 2.)
    elif (simset == "4096_adaphase_256"):
        nyquist= (pi / 656.25)*256.# (256. / 2.)
    elif (simset == "64_adaphase_1024"):
        nyquist= (pi / 656.25)*1024.# (1024. / 2.)
    elif (simset == "64_curiephase_1024"):
        nyquist= (pi / 656.25)*1024.# (1024. / 2.)
    idx = (power_k < nyquist) 
    power_k_new = power_k[idx]
    power_p_new = power_p[idx]
    return power_k_new, power_p_new
# ---------------------------------------------------------------------------- #



#------------------------------- REBIN Y(k) WITH Dk/k FIXED ------------------ #
def pkann_power(ipar = 0, dpar = 0.05, fact = 1, powertype = "power", ioutput = 1, mainpath = "", aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0)):
    
    par=np.array([0.2573*0.5184,0.04356*0.5184,0.963,-1.,0.801,0.]) #omega_m h^2, omega_b h^2, n_s, w, sigma_8, m_nu
    list_z = [99., 2., 1.5, 1., 0.7, 0.5, 0.3, 0.01, 0.]
    par[ipar]+=fact*dpar*par[ipar]
    redshift = list_z[ioutput-1]

    pfile = "pkann_spectra/pkann_"+str(ipar)+"_"+str(dpar)+"_"+str(fact)+"_"+powertype+"_"+str(ioutput)+".txt"
    if (os.path.isfile(pfile)):
        power_k,power_pkann = np.loadtxt(pfile,unpack=True)
    else:

        pkanndir="/data/lblot/deus_pur/PkANN/"
        infile1=pkanndir+"Testing_set/parameters.txt"
        infile2=pkanndir+"Testing_set/z_list.txt"
        outfile=pkanndir+"Testing_set/PkANN_predictions/model_1/ps_nl_1.dat"
        fltformat="%1.8f"
        f1 = open(infile1, "w")
        f1.write("   "+str(fltformat%par[0])+"   "+str(fltformat%par[1])+"   "+str(fltformat%par[2])+"   "+str(fltformat%par[3])+"   "+str(fltformat%par[4])+"   "+str(fltformat%par[5])+"\n")
        f1.close()
        f2 = open(infile2, "w")
        f2.write(str(fltformat%redshift)+"\n")
        f2.close()

        command="cd "+pkanndir+"; ./unix.sh > output.out; cd - > tmp"
        os.system(command)
        kpkann,ppkann = np.loadtxt(outfile,skiprows=4,unpack=True)

        if (powertype=="nyquist" or powertype=="mcorrected"):
            power_k,dummy=nyquist_power(mainpath,"4096_adaphase_256",1,ioutput,aexp,growth_a,growth_dplus)
        else:
            power_k,dummy,dummy=read_power(file_path("power", mainpath, "4096_adaphase_256", 1, ioutput))
        power_pkann = np.interp(power_k,kpkann,ppkann)

        fltformat2="%-.12e"
        fout = open(pfile, "w")
        for i in range(0,power_k.size):
            fout.write(str(fltformat2%power_k[i])+" "+str(fltformat2%power_pkann[i])+"\n")
        fout.close()

    return power_k, power_pkann
# ---------------------------------------------------------------------------- #



#------------------------------- REBIN Y(k) WITH Dk/k FIXED ------------------ #
def rebin(k= np.zeros(0),y= np.zeros(0),lim=0.1):
    delta_k = np.zeros(k.size)
    for i in range(0,k.size-1):
        delta_k[i] = k[i+1]-k[i]
    delta_k[delta_k.size-1]=delta_k[delta_k.size-2]
    
    interval=0.
    j=0
    count=0
    new_y=np.zeros(k.size)
    new_k=np.zeros(k.size)

    for i in range(0,k.size):
        interval+=delta_k[i]
        new_y[j]+=y[i]
        new_k[j]=k[i]-interval/2.
        count+=1
        if (interval/new_k[j]>=lim):
            new_y[j]/=float(count)
            count=0
            interval=0.
            j+=1
            max_i = i
        
    rebin_y = new_y[0:j]
    rebin_k = new_k[0:j]

    error=np.zeros(y.size)
    interval=0
    count=0
    j=0
    for i in range(0,max_i+1):
        interval+=delta_k[i]
        nw_k=k[i]-interval/2.
        error[j]=(y[i]-rebin_y[j])*(y[i]-rebin_y[j])
        count+=1
        if (interval/nw_k>=lim):
            if (count==1):
                error[j]=0.
            else:
                error[j]/=float(count-1)
            j+=1
            count=0
            interval=0
    error=np.sqrt(error)
    rebin_e = error[0:j]

    return rebin_k,rebin_y,rebin_e
