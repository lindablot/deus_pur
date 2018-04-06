#!/usr/bin/env python
# utilities.py - Vincent Reverdy (vince.rev@gmail.com) and Linda Blot (linda.blot@obspm.fr) - 2013
# ---------------------------------- IMPORT ---------------------------------- #
import math
import os
import numpy as np
# ---------------------------------------------------------------------------- #


# ------------------------------- SIMSET CLASS ------------------------------- #
class Simset(object):

    def __init__(self, l_box, npart, nsimmax):
        self.l_box = l_box
        self.npart = npart
        self.nsimmax = nsimmax
        self.nyquist = math.pi/self.l_box*self.npart

    def num_modes(self, power_k):
        delta_k = np.diff(power_k)
        delta_k = np.append(delta_k, delta_k[delta_k.size-1])
        return self.l_box**3/(2.*math.pi**2) * (power_k*power_k*delta_k + delta_k*delta_k*delta_k/12.)


class DeusPurSet(Simset):
    
    simsets = ["4096_furphase_256", "4096_adaphase_256", "4096_otherphase_256", "all_256", "4096_furphase_512",
               "64_adaphase_1024", "64_curiephase_1024", "all_1024", "512_adaphase_512_328-125",
               "4096_furphase", "4096_otherphase"]
    
    def __init__(self, name, nmodel=0):
        
        if name in self.simsets:
            self.name = name
        else:
            raise ValueError("Simset name not found")

        if self.name == "4096_furphase_256" or self.name == "4096_adaphase_256" or self.name == "4096_otherphase_256" \
                or self.name == "all_256":
            self.npart = 256.
            self.l_box = 656.25
            if self.name == "all_256":
                self.nsimmax = 12288
            else:
                self.nsimmax = 4096
            self.cosmo = False
        elif self.name == "4096_furphase_512":
            self.npart = 512.
            self.l_box = 1312.5
            self.nsimmax = 512
            self.cosmo = False
        elif self.name == "64_adaphase_1024" or self.name == "64_curiephase_1024" or self.name == "all_1024":
            self.npart = 1024.
            self.l_box = 656.25
            if self.name == "64_adaphase_1024":
                self.nsimmax = 64
            elif self.name == "64_curiephase_1024":
                self.nsimmax = 32
            else:
                self.nsimmax = 96
            self.cosmo = False
        elif self.name == "512_adaphase_512_328-125":
            self.npart = 512.
            self.l_box = 328.125
            self.nsimmax = 512
            self.nmodel = nmodel
            self.cosmo = True
        elif self.name == "4096_otherphase" or self.name == "4096_otherphase":
            self.npart = 4096.
            self.l_box = 10500.
            self.nsimmax = 1
            self.cosmo = False

        Simset.__init__(self, self.l_box, self.npart, self.nsimmax)
        self.nyquist = math.pi/self.l_box*self.npart
        
        if name == "all_256" or name == "all_1024":
            self.composite = True
        else:
            self.composite = False
# ---------------------------------------------------------------------------- #


# ------------------------------ SIMULATION SET ITERATOR --------------------------------- #
def sim_iterator(simset=DeusPurSet("all_256"), isim=1, random=False, replace=False):
    
    if random:
        fname = "random_series_"+simset.name+".txt"
        if os.path.isfile(fname) and not replace:
            series = np.loadtxt(fname, unpack=True)
        else:
            series = np.random.permutation(simset.nsimmax)
            f = open(fname, "w")
            for i in xrange(1, series.size):
                f.write(str("%05d" % series[i])+"\n")
            f.close()
        isim = series[isim]

    if simset.name == "all_256":
        if isim < 4097:
            true_set = "4096_furphase_256"
            true_isim = isim
        elif isim < 8193:
            true_set = "4096_adaphase_256"
            true_isim = isim - 4096
        else:
            true_set = "4096_otherphase_256"
            true_isim = isim - 8192
    elif simset.name == "all_1024":
        if isim < 65:
            true_set = "64_adaphase_1024"
            true_isim = isim
        else:
            true_set = "64_curiephase_1024"
            true_isim = isim - 64
    elif simset.name == "all_cosmo":
        true_set = "512_adaphase_512_328-125"
        true_isim = isim
    else:
        true_set = simset.name
        true_isim = isim

    return true_set, true_isim
# ---------------------------------------------------------------------------- #


# -------------------------------- INPUT FILE NAME --------------------------------- #
def input_file_name(filetype="", mainpath="", setname="", nsim=1, noutput=1, nmodel=0,
                    okprint=False, powertype="gridcic"):
    
    fullpath = str(mainpath)

    if filetype == "power":
        dataprefix = "/power/power"+powertype+"_"
    elif filetype == "info":
        dataprefix = "/info/info_"
    elif filetype == "massfunction":
        dataprefix = "/massfunction/massfunction_"
    else:
        raise ValueError("filetype not found")

    if setname == "4096_furphase":
        fullpath += setname + dataprefix + str(int(noutput)).zfill(5) + ".txt"
    elif setname == "4096_otherphase":
        fullpath += setname + dataprefix + str(int(noutput)).zfill(5) + ".txt"
    elif setname == "4096_furphase_512":
        fullpath += setname + "/boxlen1312-5_n512_lcdmw7_" + str(int(nsim)).zfill(5) + \
                    dataprefix + str(int(noutput)).zfill(5) + ".txt"
    elif setname == "4096_furphase_256":
        fullpath += setname + "/boxlen656-25_n256_lcdmw7_" + str(int(nsim)).zfill(5) + \
                    dataprefix + str(int(noutput)).zfill(5) + ".txt"
    elif setname == "4096_otherphase_256":
        fullpath += setname + "/boxlen656-25_n256_lcdmw7_" + str(int(nsim)).zfill(5) + \
                    dataprefix + str(int(noutput)).zfill(5) + ".txt"
    elif setname == "4096_adaphase_256":
        fullpath += setname + "/boxlen656-25_n256_lcdmw7_" + str(int(nsim)).zfill(5) + \
                    dataprefix + str(int(noutput)).zfill(5) + ".txt"
    elif setname == "64_adaphase_1024":
        fullpath += setname + "/boxlen656-25_n1024_lcdmw7_" + str(int(nsim)).zfill(5) + \
                    dataprefix + str(int(noutput)).zfill(5) + ".txt"
    elif setname == "64_curiephase_1024":
        fullpath += setname + "/boxlen656-25_n1024_lcdmw7_" + str(int(nsim)).zfill(5) + \
                    dataprefix + str(int(noutput)).zfill(5) + ".txt"
    elif setname == "512_adaphase_512_328-125":
        fullpath += setname + "/boxlen328-125_n512_model" + str(int(nmodel)).zfill(5) + "_" + str(int(nsim)).zfill(5) \
                    + dataprefix + str(int(noutput)).zfill(5) + ".txt"
    else:
        raise ValueError("setname not found")
    
    if okprint:
        print fullpath

    return fullpath
# ---------------------------------------------------------------------------- #


# --------------------------- OUTPUT FILE NAME --------------------------- #
def output_file_name(prefix="cov", powertype="", simset=DeusPurSet("all_256"),
                     isimmin=1, isimmax=1, ioutput=1, nmodel=0):

    nsim = isimmax-isimmin
    
    fname = prefix+"_"+powertype+"_"+str("%05d" % ioutput)+"_"
    if nsim == simset.nsimmax:
        if simset.cosmo:
            fname = fname+"cosmo_model"+str(int(nmodel))+".txt"
        else:
            fname = fname+simset.name+".txt"
    else:
        if simset.cosmo:
            fname = fname+"cosmo_model"+str(int(nmodel))+"_"+str(isimmin)+"_"+str(isimmax)+".txt"
        else:
            fname = fname+simset.name+"_"+str(isimmin)+"_"+str(isimmax)+".txt"

    return fname
# ---------------------------------------------------------------------------- #


# ------------------------------- EXTRAPOLATE -------------------------------- #
def extrapolate(value_x, array_x, array_y):
    if value_x < array_x[0]:
        value_y = array_y[0]+(value_x-array_x[0])*(array_y[0]-array_y[1])/(array_x[0]-array_x[1])
    elif value_x > array_x[-1]:
        value_y = array_y[-1]+(value_x-array_x[-1])*(array_y[-1]-array_y[-2])/(array_x[-1]-array_x[-2])
    else:
        value_y = np.interp(value_x, array_x, array_y)
    return value_y
# ---------------------------------------------------------------------------- #


# ------------------------------- REBIN Y(k) WITH Dk/k FIXED ------------------ #
def rebin(k=np.zeros(0), y=np.zeros(0), lim=0.1):
    
    delta_k = np.diff(k)
    delta_k = np.append(delta_k, delta_k[delta_k.size-1])
    
    interval = 0.
    j = 0
    count = 0
    new_y = np.zeros(k.size)
    new_k = np.zeros(k.size)

    for i in xrange(0, k.size):
        interval += delta_k[i]
        new_y[j] += y[i]
        new_k[j] = k[i]-interval/2.
        count += 1
        if interval/new_k[j] >= lim:
            new_y[j] /= float(count)
            count = 0
            interval = 0.
            j += 1
            max_i = i
        
    rebin_y = new_y[0:j]
    rebin_k = new_k[0:j]

    error = np.zeros(y.size)
    interval = 0
    count = 0
    j = 0
    for i in xrange(0, max_i+1):
        interval += delta_k[i]
        nw_k = k[i]-interval/2.
        error[j] = (y[i]-rebin_y[j])*(y[i]-rebin_y[j])
        count += 1
        if interval/nw_k >= lim:
            if count == 1:
                error[j] = 0.
            else:
                error[j] /= float(count-1)
            j += 1
            count = 0
            interval = 0
    error = np.sqrt(error)
    rebin_e = error[0:j]

    return rebin_k, rebin_y, rebin_e
# ---------------------------------------------------------------------------- #


# ----------------------  REBIN POWER SPECTRUM EXACTLY ---------------------- #
def rebin_pk(k, pk, nk, nbins):
    if k.size % nbins != 0:
        k = k[:(k.size-k.size % nbins)]
        nk = nk[:(k.size-k.size % nbins)]
        pk = pk[:(k.size-k.size % nbins)]
    
    k_new = k.reshape(-1, nbins)
    k_new = np.mean(k_new, axis=1)
    pknk = pk*nk
    pknk_new = np.sum(pknk.reshape(-1, nbins), axis=1)
    nksum = np.sum(nk.reshape(-1, nbins), axis=1)
    pk_new = pknk_new/nksum

    return k_new, pk_new
# ---------------------------------------------------------------------------- #
