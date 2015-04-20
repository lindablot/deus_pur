#!/usr/bin/env python
# power_types.py - Vincent Reverdy (vince.rev@gmail.com) and Linda Blot (linda.blot@obspm.fr) - 2013
# ---------------------------------- IMPORT ---------------------------------- #
import math
import glob
import os
import sys
import numpy as np
from read_files import *
from utilities import *
import scipy.interpolate as itl
# ---------------------------------------------------------------------------- #



# --------------------------- RENORMALIZED POWER SPECTRUM ----------------------------- #
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



# ----------------------------- POWER SPECTRUM CORRECTED FOR a DIFFERENCES ------------------------------ #
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



# ----------------------------- POWER SPECTRUM CUT AT NYQUIST FREQUENCY ------------------------------ #
def nyquist_power(mainpath = "", simset = "", nsim = 1, noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0)):
    pi = math.pi
    power_k, power_p_raw, dummy = read_power(file_path("power", mainpath, simset, nsim, noutput))
    if (aexp != 0.):
        aexp_raw = read_info(file_path("info", mainpath, simset, nsim, noutput))
        dplus_raw = extrapolate([aexp_raw], growth_a, growth_dplus)
        dplus = extrapolate([aexp], growth_a, growth_dplus)
        power_p = (power_p_raw*dplus*dplus)/(dplus_raw*dplus_raw)
    else:
        power_p = power_p_raw
    if (simset == "4096_furphase"):
        nyquist = (pi / 10000.) * 4096.
    elif (simset == "4096_otherphase"):
        nyquist= (pi / 10000.)* 4096.
    elif (simset == "4096_furphase_512"):
        nyquist= (pi / 1312.5)*512.
    elif (simset == "4096_furphase_256"):
        nyquist= (pi / 656.25)*256.
    elif (simset == "4096_otherphase_256"):
        nyquist= (pi / 656.25)*256.
    elif (simset == "4096_adaphase_256"):
        nyquist= (pi / 656.25)*256.
    elif (simset == "64_adaphase_1024"):
        nyquist= (pi / 656.25)*1024.
    elif (simset == "64_curiephase_1024"):
        nyquist= (pi / 656.25)*1024.
    idx = (power_k < nyquist) 
    power_k_new = power_k[idx]
    power_p_new = power_p[idx]
    return power_k_new, power_p_new
# ---------------------------------------------------------------------------- #



#------------------------------- POWER SPECTRUM COMPUTED BY PkANN ------------------ #
def pkann_power(ipar = 0, dpar = 0.05, fact = 1, powertype = "power", ioutput = 1, mainpath = "", aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0)):
    
    par=np.array([0.2573*0.5184,0.04356*0.5184,0.963,-1.,0.801,0.,1.]) #omega_m h^2, omega_b h^2, n_s, w, sigma_8, m_nu, bias
    list_z = [99., 2., 1.5, 1., 0.7, 0.5, 0.3, 0.01, 0.]
    par[ipar]+=fact*dpar*abs(par[ipar])
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
        power_pkann = par[6]*par[6]*np.interp(power_k,kpkann,ppkann)

        fltformat2="%-.12e"
        fout = open(pfile, "w")
        for i in xrange(0,power_k.size):
            fout.write(str(fltformat2%power_k[i])+" "+str(fltformat2%power_pkann[i])+"\n")
        fout.close()

    return power_k, power_pkann
# ---------------------------------------------------------------------------- #

