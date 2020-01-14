#!/usr/bin/env python
# Linda Blot (linda.blot@obspm.fr) - 2019
# ---------------------------------- IMPORT ---------------------------------- #
from power_stats import *
# ---------------------------------------------------------------------------- #


# -------------------------------- MEAN CLK ---------------------------------- #
def mean_clk(powertype, mainpath, simset, isimmin, isimmax, zsource, rebin=0, okprint=False, store=False, outpath="."):
    """ Mean and standard deviation of convergence Cls. See power_spectrum for the description of the power spectrum types. If file exists it will be read from file.

    
    Parameters
    ---------
    powertype: string
        type of power spectrum: power, nyquist, renormalized, corrected, mcorrected, linear or linear_mock
    mainpath: string
        path to base folder
    simset: Simset instance
        simulation set
    isimmin: int
        mimimum simulation number
    isimmax: int
        maximum simulation number
    zsource: float
        redshift of the source
    okprint: bool
        verbose (default False)
    store: bool
        store file. If True and file exists it will be overwritten (default False)
    outpath: string
        path where output file is stored (default empty)
    
    Returns
    -------
    3 numpy arrays
        l, average Cl of convergence and standard deviation
    """

    nsim = isimmax-isimmin+1
    fname = outpath+"/"+output_file_name("mean_cls", powertype, simset, isimmin, isimmax, zsource)

    if os.path.isfile(fname) and not store:
        if okprint:
            print "Reading mean Cls from file: ", fname
        l, clk_mean, clk_sigma = np.loadtxt(fname, unpack=True)
    else:
        if okprint:
            print "Computing mean and standard deviation of Cls"
        l, clks = load_clk(powertype, mainpath, simset, zsource, rebin, outpath, okprint)
        clks = clks[isimmin-1:isimmax]

        clk_mean = np.mean(clks, axis=0)
        if nsim > 1:
            clk_sigma = np.std(clks, axis=0, ddof=1)
        else:
            if okprint:
                print "Standard deviation not computed because there is only one simulation"
            clk_sigma = np.zeros(l.size)

        if store:
            if okprint:
                print "Writing file: ", fname
            f = open(fname, "w")
            for i in xrange(0, l.size):
                f.write(str("%-.12e" % l[i])+" "+str("%-.12e"%clk_mean[i])+" " +
                        str("%-.12e" % clk_sigma[i])+"\n")
            f.close()

    return l, clk_mean, clk_sigma
# ---------------------------------------------------------------------------- #



# ----------------------- PDF OF CONVERGENCE CLS ----------------------------- #
def distrib_clk(powertype, mainpath, simset, isimmin, isimmax, zsource, nbin=50, lref=50, rebin=0, norm=False, okprint=False, store=False, outpath="."):
    """ Distribution of convergence Cl at given l. See power_spectrum for the description of the power spectrum types. If file exists it will be read from file.

    
    Parameters
    ---------
    powertype: string
        type of power spectrum: power, nyquist, renormalized, corrected, mcorrected, linear or linear_mock
    mainpath: string
        path to base folder
    simset: Simset instance
        simulation set
    isimmin: int
        mimimum simulation number
    isimmax: int
        maximum simulation number
    zsource: float
        redshift of the source
    nbin: int
        number of P(k) bins (default 50)
    lref: float
        l value (default 50)
    norm: bool
        if True returns probability density, if False returns the frequency (default False)
    okprint: bool
        verbose (default False)
    store: bool
        store file. If True and file exists it will be overwritten (default False)
    outpath: string
        path where output file is stored (default empty)
    
    Returns
    -------
    numpy array
        Cl bin center
    numpy array
        frequency or probability density
    """

    nsim = isimmax - isimmin+1
    clk_values = np.zeros(nsim)
    fname = outpath+"/"+output_file_name("distrib_clk_l"+str(lref), powertype, simset, isimmin, isimmax, zsource)
    if os.path.isfile(fname) and not store:
        if okprint:
            print "Reading distribution from file ", fname
        bincenter, nclk_bin = np.loadtxt(fname, unpack=True)
    else:
        if okprint:
            print "Computing distribution at l=",lref
        l, clks = load_clk(powertype, mainpath, simset, zsource, rebin)
        clk_values = clks[isimmin-1:isimmax,np.searchsorted(l, lref)]

        nclk_bin, bins = np.histogram(clk_values, nbin, density=norm)
        bincenter = 0.5*(bins[1:]+bins[:-1])
        
        if store:
            if okprint:
                print "Writing file: ", fname
            f = open(fname, "w")
            for i in xrange(0, bincenter.size):
                f.write(str("%-.12e" % bincenter[i])+" "+str("%-.12e" % nclk_bin[i])+"\n")
            f.close()

        return bincenter, nclk_bin
# ---------------------------------------------------------------------------- #


# ------------------ HIGH MOMENTS OF CLK PDF ---------------------- #
def high_moments_clk(powertype, mainpath, simset, isimmin, isimmax, zsource, rebin=0, unbiased=True, okprint=False, store=False, outpath="."):
    """ Skewness and Kurtosis of the distribution of convergence Cls. See power_spectrum for the description of the power spectrum types. If file exists it will be read from file.

    
    Parameters
    ---------
    powertype: string
        type of power spectrum: power, nyquist, renormalized, corrected, mcorrected, linear or linear_mock
    mainpath: string
        path to base folder
    simset: Simset instance
        simulation set
    isimmin: int
        mimimum simulation number
    isimmax: int
        maximum simulation number
    zsource: float
        redshift of the source
    unbiased: bool
        use unbiased estimators for Skewness and Kurtosis (default True)
    okprint: bool
        verbose (default False)
    store: bool
        store file. If True and file exists it will be overwritten (default False)
    outpath: string
        path where output file is stored (default empty)
    
    Returns
    -------
    3 numpy arrays
        l, skewness and kurtosis
    """

    nsim = isimmax-isimmin+1
    
    if unbiased:
        bias = "unbiased"
    else:
        bias = "biased"

    fname = outpath+"/"+output_file_name("high_moments_clk_"+bias, powertype, simset, isimmin, isimmax, zsource)
    if os.path.isfile(fname) and not store:
        if okprint:
            print "Reading skewness and kurtosis from file ", fname
        l, clk_skew, clk_kurt = np.loadtxt(fname, unpack=True)
    else:
        if okprint:
            print "Computing skewness and kurtosis with ", bias, " estimator"
        l, clk_mean, clk_sigma = mean_clk(powertype, mainpath, simset, isimmin, isimmax, zsource, rebin)
        clk_skew = np.zeros(l.size)
        clk_kurt = np.zeros(l.size)
        ll, clks = load_clk(powertype, mainpath, simset, zsource, rebin, outpath)
        if l.size!=ll.size:
            print "Inconsistent l size in high moments"
            exit()
        for isim in xrange(isimmin, isimmax+1):
            clk_skew += (clks[isim-1]-clk_mean)*(clks[isim-1]-clk_mean)*(clks[isim-1]-clk_mean)
            clk_kurt += (clks[isim-1]-clk_mean)*(clks[isim-1]-clk_mean)*(clks[isim-1]-clk_mean)*(clks[isim-1]-clk_mean)

        if unbiased:
            clk_skew *= float(nsim)/(float(nsim-1)*float(nsim-2))
            clk_skew /= clk_sigma * clk_sigma * clk_sigma
            clk_kurt *= float(nsim+1) * float(nsim) / (float(nsim-1) * float(nsim-2) * float(nsim-3) *
                                                         clk_sigma * clk_sigma * clk_sigma * clk_sigma)
            clk_kurt -= 3. * float(nsim-1) * float(nsim-1) / (float(nsim-2)*float(nsim-3))
        else:
            clk_skew /= float(nsim)
            clk_skew /= clk_sigma * clk_sigma * clk_sigma
            clk_kurt /= float(nsim)
            clk_kurt /= clk_sigma * clk_sigma * clk_sigma * clk_sigma
            clk_kurt -= 3.

        if store:
            if okprint:
                print "Writing file: ", fname
            f = open(fname, "w")
            for i in xrange(0, l.size):
                f.write(str("%-.12e" % l[i]) + " " + str("%-.12e" % clk_skew[i]) + " " +
                        str("%-.12e" % clk_kurt[i]) + "\n")
            f.close()
    
    return l, clk_skew, clk_kurt
# ---------------------------------------------------------------------------- #



# --------------------------- COMPUTE CLS FROM PKS --------------------------- #
def pk2clk(powertype, mainpath, simset, nsim, zsource, rebin=0, k=np.zeros(0), pk_allz=np.zeros(0), lmax=4000):
    """ Function to compute Cls of convergence summing the power spectra of all the snapshots between the observer and the source redshift
        
    Parameters
    ---------
    powertype: string
        type of power spectrum (power, nyquist, renormalized, corrected, mcorrected, linear, linear_mock)
    mainpath: string
        path to base folder
    simset: Simset instance
        simulation set
    nsim: int
        simulation number
    zsource: float
        redshift of the source
    k: numpy array
        k (optional)
    pk_allz: numpy array (nsnapshot, k.size)
        array of P(k) for all snapshots (optional)
    lmax: int
        maximum l (optional, default 4000), it is overridden based on kmax of pk
        
    Returns
    -------
    numpy array
        l
    numpy array
        Cl
    """
    
    read_pk_file = k.size==0
    zlist = 1./np.array(simset.alist)-1.
    snaplist = np.arange(1,zlist.size+1)
    # initialise cosmology
    import astropy.cosmology as cosmology
    Omega_m = simset.cosmo_par['om_m']/simset.cosmo_par['h']**2
    cosmo = cosmology.FlatwCDM(100.,Omega_m,simset.cosmo_par['w_0'])
    # compute comoving distance to source and snapshot centers
    rs = cosmo.comoving_distance(zsource).value
    rlist = cosmo.comoving_distance(zlist).value
    # select snapshots between source and observer
    zindex = ((zlist>0.) & (zlist<zsource))
    zlist = zlist[zindex]
    snaplist = snaplist[zindex]
    # compute pre-factor for weights
    clight = 299792.458
    fact = 100./clight
    # initialise array of cls
    l=np.arange(0,lmax+1)
    nslice = zlist.size
    clarr=np.zeros((nslice,l.size))
    for i in range(nslice):
        rmed = cosmo.comoving_distance(zlist[i]).value
        if i==0:
            r2=rs
        else:
            r2=(rlist[snaplist[i]-2]+rlist[snaplist[i]-1])/2.
        r1=(rlist[snaplist[i]-1]+rlist[snaplist[i]])/2.
        drl = r2-r1
        a = 1./(zlist[i]+1.)
        kmax=1.05*lmax/rmed
        if read_pk_file:
            k, pk = power_spectrum(powertype, mainpath, simset, nsim, snaplist[i], a)
        else:
            pk = pk_allz[snaplist[i]-1]
        if kmax>k[-1]:
            l_max = k[-1]*rmed
        else:
            l_max = lmax
        pk=np.extract(k<=kmax,pk)
        if zlist[i]==np.max(zlist):
            l_min = k[0]*rmed
        weight=1.5*omega_m*fact*fact*(rs-rmed)/(rs*a)
        li=np.extract(k<=kmax,k)*rmed
        limin=int(np.min(li))
        nli=int(np.max(li))-limin+1
        if nli==0:
            continue
        lbi=np.arange(limin,limin+nli)
        cli=drl*weight*weight*pk
        clbi=np.interp(lbi,li,cli)
        index=((lbi>0) & (lbi<=lmax))
        lbi=lbi[index]
        clbi=clbi[index]
        clarr[i,lbi]=clbi
    cltot=np.sum(clarr,axis=0)
    index=((l>l_min) & (l<=l_max))
    l = l[index]
    cltot = cltot[index]
    return l,cltot
# ---------------------------------------------------------------------------- #



# ----------------------------- LOAD ALL CLS --------------------------------- #
def load_clk(powertype, mainpath, simset, zsource, rebin=0, okprint=False, load_power_file=True, outpath="."):
    """
    Load all the convergence Cls of a given simulation set in memory
    
    Parameters
    ----------
    powertype: string
        type of power spectrum (power, nyquist, renormalized, corrected, mcorrected, linear, linear_mock)
    mainpath: string
        path to base folder
    simset: Simset instance
        simulation set
    zsource: float
        redshift of the source
    rebin: int
        number of bins to combine when rebinning (default 0, i.e. no rebinning)
    okprint: bool
        verbose (default False)
    load_power_file: bool
        use function load_power to load all the power spectra of the set (faster but more memory consuming, default True)
    outpath: string
        path where output file is stored (default .)
        
    Returns
    -------
    2 numpy arrays
        vector of k values and array of power spectra of shape (nsim,nbin)
    """
    fname = outpath+"/"+output_file_name("clks", powertype, simset, 1, simset.nsimmax, zsource, extension=".npy")
    if os.path.isfile(fname):
        l, dummy = pk2clk(powertype, mainpath, simset, 1, zsource)
        clks_array=np.load(fname)
    else:
        if load_power_file:
            power_array = []
            for isnap in range(1,len(simset.alist)+1):
                power_k, power_ps = load_power(powertype, mainpath, simset, isnap, simset.snap_to_a(isnap), rebin=0, outpath=outpath, okprint=True)
                power_array.append(power_ps)
            power_array=np.array(power_array)
            l, dummy = pk2clk(powertype, mainpath, simset, 1, zsource, 0, power_k, power_array[:,0,:])
            clks_array = np.zeros((simset.nsimmax, l.size))
            for isim in xrange(0, simset.nsimmax):
                if okprint:
                    print simset.name, isim+1
                dummy, clks_array[isim] = pk2clk(powertype, mainpath, simset, isim+1, zsource, 0, power_k, power_array[:,isim,:])
        else:
            l, dummy = pk2clk(powertype, mainpath, simset, 1, zsource)
            clks_array = np.zeros((simset.nsimmax, l.size))
            for isim in xrange(0, simset.nsimmax):
                if okprint:
                    print simset.name, isim+1
                dummy, clks_array[isim] = pk2clk(powertype, mainpath, simset, isim+1, zsource)
        np.save(fname, clks_array)
    if rebin>0:
        l, clks_array = rebin_cl(l, clks_array, rebin)
    return l, clks_array
# ---------------------------------------------------------------------------- #
