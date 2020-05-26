#!/usr/bin/env python
# utilities.py - Vincent Reverdy (vince.rev@gmail.com) and Linda Blot (linda.blot@obspm.fr) - 2013
# ---------------------------------- IMPORT ---------------------------------- #
import math
import os
import numpy as np
import pandas as pd
# ---------------------------------------------------------------------------- #


# ------------------------------- SIMSET CLASS ------------------------------- #
class Simset(object):
    """
    Class that represents the simulation set

    Attributes
    ---------
    l_box : double
        size of the simulation box
    npart : double
        cube root of the number of particles
    nsimmax : int
        last simulation number
    nyquist : double
        half of the nyquist frequency of the power spectrum grid
    cosmo_par : dictionary with keys om_b, om_m, n_s, h, w_0, w_a, sigma_8, m_nu, om_nu
        cosmological parameters Omega_m*h^2, Omega_m*h^2, n_s, h, w_0, w_a, sigma_8, m_nu, Omega_nu*h^2

    Methods
    ------
    num_modes(k)
        Compute the number of modes at given k values
    """

    def __init__(self, l_box, npart, nsimmax, cosmo_par):
        """
        Parameters
        ----------
        l_box : double
            size of the simulation box
        npart : double
            cube root of the number of particles
        nsimmax : int
            last simulation number
        cosmo_par : dictionary with keys om_b, om_m, n_s, h, w_0, w_a, sigma_8, m_nu, om_nu
            cosmological parameters Omega_m*h^2, Omega_m*h^2, n_s, h, w_0, w_a, sigma_8, m_nu, Omega_nu*h^2
        """

        self.l_box = l_box
        self.npart = npart
        self.nsimmax = nsimmax
        self.nyquist = math.pi/self.l_box*self.npart
        self.cosmo_par = cosmo_par

    def num_modes(self, power_k):
        """Computes the number of modes at given k values

        Assumes regular linear binning

        Parameters
        ----------
        power_k : numpy array
            array of k values

        Returns
        -------
        numpy array
            number of modes at given k values
        """

        delta_k = np.diff(power_k)
        delta_k = np.append(delta_k, delta_k[-1])
        return self.l_box**3 / (2.*math.pi**2) * (power_k*power_k*delta_k + delta_k*delta_k*delta_k/12.)


class DeusPurSet(Simset):
    """ Derived class from the Simset class used to represent the Deus Pur simulation set

    Given the set name sets all attributes of the Simset class and adds some useful attributes

    Attributes
    ----------
    name : string
        simset name
    cosmo : bool
        multiple cosmology set
    composite : bool
        combination of different simulation sets
    alist : list
        list of expansion factor values of snapshots
    model : string
        cosmological model name
    nmodel : int
        cosmological model number

    Methods
    -------
    snap_to_a(nsnap)
        gives the expansion factor corresponding to the snapshot number
    snap_to_z(nsnap)
        gives the redshift corresponding to the snapshot number
    """

    simsets = ["4096_furphase_256", "4096_adaphase_256", "4096_otherphase_256", "all_256", "4096_furphase_512",
               "64_adaphase_1024", "64_curiephase_1024", "all_1024", "512_adaphase_512_328-125",
               "4096_furphase", "4096_otherphase"]

    def __init__(self, name, nmodel=0, datapath="/data/deus_pur_cosmo/data/"):
        """
        Parameters
        ----------
        name: string
            name of the simulation set
        nmodel: int, optional
            number of the cosmological model (default is 0)
        datapath: string, optional
            path to the folder that contains the models_parameters.txt file (default is /data/deus_pur_cosmo/data/)
        """

        self.datapath = datapath
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
            self.alist=[0.01,0.33,0.4,0.5,0.59,0.66,0.77,0.91,1.]
            self.cosmo = False
        elif self.name == "4096_furphase_512":
            self.npart = 512.
            self.l_box = 1312.5
            self.nsimmax = 512
            self.alist = None
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
            self.alist=[0.05,0.33,0.4,0.5,0.59,0.66,0.77,0.91,1.]
            self.cosmo = False
        elif self.name == "512_adaphase_512_328-125":
            self.npart = 512.
            self.l_box = 328.125
            self.nsimmax = 512
            self.alist=[0.0097153,0.3333,0.3650,0.4000,0.4167,0.4444,0.4762, 0.5000,0.5263, 0.5556, 0.5882,0.6250, 0.6667,0.7042, 0.7692,0.800,0.8696, 0.9091,0.9524,1.0000]
            self.nmodel = nmodel
            self.cosmo = True
        elif self.name == "4096_otherphase" or self.name == "4096_otherphase":
            self.npart = 4096.
            self.l_box = 10500.
            self.nsimmax = 1
            self.alist = None
            self.cosmo = False

        if not self.cosmo or nmodel==0:
            self.cosmo_par={'om_b': 0.04356*0.5184, 'om_m': 0.2573*0.5184, 'n_s': 0.963, 'h': 0.72, 'w_0': -1., 'w_a':0., 'sigma_8': 0.801, 'm_nu': 0., 'om_nu':0.}
        else:
            Om_b, Om_m, Om_lr, Om_nu, h, n_s, w, sigma_8 = np.genfromtxt(datapath+"/models_parameters.txt",unpack=True,skip_header=1)
            sigma_8_camb = np.array([1.5392379, 0.79753089, 0.8379174, 0.7400835, 0.7975309, 0.7975309, 0.9173459, 0.6546858, 0.6992297, 0.8966455])
            self.cosmo_par = {'om_b': Om_b[nmodel-1]*h[nmodel-1]**2, 'om_m': Om_m[nmodel-1]*h[nmodel-1]**2, 'n_s': n_s[nmodel-1], 'h': h[nmodel-1], 'w_0': w[nmodel-1], 'w_a':0., 'sigma_8': sigma_8[nmodel-1], 'm_nu': 0., 'om_nu': 0., 'sigma_8_camb': sigma_8_camb[nmodel-1]}

        if self.cosmo:
            self.model = "model"+str(int(nmodel)).zfill(5)
        else:
            self.model = "lcdmw7"
        self.nmodel = nmodel

        Simset.__init__(self, self.l_box, self.npart, self.nsimmax, self.cosmo_par)
        self.nyquist = math.pi/self.l_box*self.npart

        if name == "all_256" or name == "all_1024":
            self.composite = True
        else:
            self.composite = False

    def snap_to_a(self, noutput):
        """Gives the expansion factor corresponding to the snapshot number

        Parameters
        ----------
        noutput : int
            snapshot number

        Returns
        -------
        double
            expansion factor corresponding to the snapshot number
        """

        if self.alist==None:
            raise NotImplementedError("alist not implemented")
        return self.alist[noutput-1]

    def snap_to_z(self, noutput):
        """Gives the redshift corresponding to the snapshot number

        Parameters
        ----------
        noutput : int
            snapshot number

        Returns
        -------
        double
            redshift corresponding to the snapshot number
        """

        return 1./self.snap_to_a(noutput)-1.

# ---------------------------------------------------------------------------- #


# ------------------------------ SIMULATION SET ITERATOR --------------------------------- #
def sim_iterator(simset=DeusPurSet("all_256"), isim=1, random=False, replace=False):
    """ Tool to iterate through composite simulation sets

    Given the composite set simulation number returns the original simulation set name and number for file retrieving. The simulations are iterated in order except if the random parameter is set to True, in this case a file with a random order is saved to disk. To replace the random order set the parameter replace to True.

    Parameters
    ---------
    simset : Simset instance
        simulation set (default DeusPurSet("all_256"))
    isim : int
        simulation number of the composite set
    random : bool
        random order of simulations
    replace : bool
        replace the random sequence of simulation numbers

    Returns
    -------
    string
        set name of the simulation
    int
        simulation number in the original set order
    """

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



# ---------------------- COSMOLOGICAL MODEL ITERATOR ------------------------- #
def cosmo_iterator(param,var):
    """
    Given a parameter name and variation returns the number of the corresponding model

    Parameters
    ----------
    param: string
        simset.cosmo_par key
    var: string
        variation of the parameter ('p'=plus, 'fid'=fiducial, 'm'=minus)

    Returns
    -------
    int
        number of cosmological model
    """

    ncosmo_map = {'om_m':{'m':8, 'fid':2, 'p':7}, 'h':{'m':9, 'fid':2, 'p':10}, 'sigma_8':{'m':5, 'fid':2, 'p':6}, 'w_0':{'m':3, 'fid':2, 'p':4}}
    return ncosmo_map[param][var]
# ---------------------------------------------------------------------------- #



# -------------------------------- INPUT FILE NAME --------------------------------- #
def input_file_name(filetype="", mainpath="", simset=DeusPurSet("all_256"), nsim=1, noutput=1,
                    okprint=False, powertype="gridcic"):
    """ Input file name generator

    Parameters
    ---------
    filetype : string
        type of file (options are power, info, massfunction)
    mainpath : string
        path to base folder
    simset : Simset instance (default is DeusPurSet("all_256"))
        simulation set
    nsim : int
        simulation number (default is 1)
    noutput : int
        snapshot number (default is 1)
    okprint : bool
        verbose mode (default is False)
    powertype : string
        type of power spectrum file (default is gridcic)

    Returns
    -------
    string
        filename
    """

    fullpath = str(mainpath)

    nmodel = simset.nmodel
    if simset.cosmo:
        if filetype == "power":
            dataprefix = "/power/output_"+str(int(noutput)).zfill(5)+"/power"+powertype+"_"
        elif filetype == "info":
            dataprefix = "/info/info_"
        else:
            raise ValueError("filetype not found")
        if simset.name == "512_adaphase_512_328-125":
            fullpath += simset.name + "/boxlen328-125_n512_model" + str(int(nmodel)).zfill(5) + "_" + str(int(nsim)).zfill(5) \
                + "/post/" + dataprefix + str(int(noutput)).zfill(5) + ".txt"
        else:
            raise ValueError("setname not found")
    else:
        if filetype == "power":
            dataprefix = "/power/power"+powertype+"_"
        elif filetype == "info":
            dataprefix = "/info/info_"
        elif filetype == "massfunction":
            dataprefix = "/massfunction/massfunction_"
        else:
            raise ValueError("filetype not found")

        setname, nsim = sim_iterator(simset, nsim)
        if setname == "4096_furphase" or setname == "4096_otherphase":
            fullpath += setname + dataprefix + str(int(noutput)).zfill(5) + ".txt"
        elif setname == "4096_furphase_512":
            fullpath += setname + "/boxlen1312-5_n512_lcdmw7_" + str(int(nsim)).zfill(5) + \
                dataprefix + str(int(noutput)).zfill(5) + ".txt"
        elif setname == "4096_furphase_256" or setname == "4096_otherphase_256" or setname == "4096_adaphase_256":
            fullpath += setname + "/boxlen656-25_n256_lcdmw7_" + str(int(nsim)).zfill(5) + \
                dataprefix + str(int(noutput)).zfill(5) + ".txt"
        elif setname == "64_adaphase_1024" or setname == "64_curiephase_1024":
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
                     isimmin=1, isimmax=1, file_id=1, extension=".txt"):
    """ Output file name generator

    Parameters
    ---------
    prefix : string
        prefix indicating the content of the file (default is cov)
    powertype : string
        type of power spectrum file (default is empty)
    simset : DeusPurSet instance
        simulation set (default is all_256)
    isimmin :  int
        initial number of simulation used (default is 1)
    isimmax : int
        final number of simulation used (default is 1)
    file_id : int or float
        number identifying the file (could be snapshot number or source redshift, default is 1)
    extension: string
        extension of the file (default ".txt")

    Returns
    ------
    string
        file name
    """

    nsim = isimmax-isimmin
    nmodel = simset.nmodel
    if isinstance(file_id, (int, long)):
        fname = prefix+"_"+powertype+"_"+str("%05d" % file_id)+"_"
    else:
        fname = prefix+"_"+powertype+"_"+str(file_id)+"_"
    if nsim == simset.nsimmax:
        if simset.cosmo:
            fname = fname+"cosmo_model"+str(int(nmodel)).zfill(2)+extension
        else:
            fname = fname+simset.name+extension
    else:
        if simset.cosmo:
            fname = fname+"cosmo_model"+str(int(nmodel)).zfill(2)+"_"+str(isimmin)+"_"+str(isimmax)+extension
        else:
            fname = fname+simset.name+"_"+str(isimmin)+"_"+str(isimmax)+extension

    return fname
# ---------------------------------------------------------------------------- #


# ------------------------------- EXTRAPOLATE -------------------------------- #
def extrapolate(value_x, array_x, array_y):
    """ Linearly extrapolate a function at a given value

    Parameters
    ----------
    value_x : double
        value to which the function is extrapolated
    array_x : numpy array
        x values of the function
    array_y : numpy array
        y values of the function

    Returns
    -------
    double
        extrapolated value
    """

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
    """ Rebin function with fixed dx/x

    Parameters
    ---------
    k : numpy array
        x values of the function
    y : numpy array
        y values of the function
    lim : double
        limit

    Returns
    ------
    double
        rebinned x
    double
        rebinned y
    double
        rebinned y error
    """

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
    """ Rebin the power spectrum exactly using the number of modes

    Parameters
    ---------
    k : numpy array
        k values of the power spectrum
    pk : numpy array
        values of the power spectrum
    nk : numpy array
        number of modes at k values
    nbins : int
        number of bins to combine

    Returns
    ------
    numpy array
        rebinned k values
    numpy array
        rebinned power spectrum values
    """

    if k.size % nbins != 0:
        k = k[:(k.size-k.size % nbins)]
        nk = nk[:(k.size-k.size % nbins)]
        pk = np.take(pk,np.arange(0,k.size-k.size % nbins),-1)
    k_new = k.reshape(-1, nbins)
    k_new = np.mean(k_new, axis=1)
    pknk = np.multiply(pk,nk)
    if pk.ndim==1:
        pknk_new = np.sum(pknk.reshape(-1, nbins), axis=1)
    else:
        pknk_new = np.sum(pknk.reshape(pknk.shape[0],pknk.shape[1]/nbins, nbins), axis=-1)
    nksum = np.sum(nk.reshape(-1, nbins), axis=1)
    pk_new = pknk_new/nksum

    return k_new, pk_new
# ---------------------------------------------------------------------------- #



# ---------------------------------------------------------------------------- #
def rebin_cl(l, Cl, nbins):
    """ Rebin the Cls

    Parameters
    ---------
    l : numpy array
        l values
    Cl : numpy array
        values of the Cls
    nbins : int
        number of bins to combine

    Returns
    ------
    numpy array
        rebinned l values
    numpy array
        rebinned Cl values
    """

    if l.size % nbins != 0:
        l = l[:(l.size-l.size % nbins)]
        Cl = np.take(Cl,np.arange(0,l.size-l.size % nbins),-1)
    l_new = l.reshape(-1, nbins)
    l_new = np.mean(l_new, axis=1)
    if Cl.ndim==1:
        Cl_new = Cl.reshape(-1, nbins)
        Cl_new = np.mean(Cl_new, axis=1)
    else:
        Cl_new = np.mean(Cl.reshape(Cl.shape[0], Cl.shape[1]/nbins, nbins), axis=-1)
    return l_new, Cl_new
# ---------------------------------------------------------------------------- #
