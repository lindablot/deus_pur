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
from power_covariance import *
from math import sqrt as sqrt
# ---------------------------------------------------------------------------- #

# -------------------------------- FISHER -------------------------------- #
def fisher_matrix(powertype = "power", galaxy = 0, list_par = [0,2,3,4], mainpath = "", simset = "", isimmin = 1, isimmax = 2, list_noutput = [1], list_aexp = [0.], growth_a = np.zeros(0), growth_dplus = np.zeros(0), okprint = False):

    nsim=isimmax-isimmin
    for i in xrange(0,len(list_aexp)):
        ioutput=list_noutput[i]
        aexp=list_aexp[i]

        covfile = "tmp/cov_"+str(nsim)+"_"+powertype+"_"+str(ioutput)+"_"+str(galaxy)+".txt"
        if (os.path.isfile(covfile)):
            biased_cov=np.loadtxt(covfile,unpack=True)
            power_k,dummy=power_spectrum(powertype,mainpath,simset,1,ioutput,aexp,growth_a,growth_dplus,False)
        else:
            power_k,power_pmean,power_psigma,power_pcov=cov_power(powertype,mainpath,simset,isimmin,isimmax,ioutput,aexp,growth_a,growth_dplus)
            biased_cov=power_pcov

            if (galaxy==1):
                bias=galaxy_bias(power_k,ioutput)
                ng=galaxy_density()
                for ik in xrange(0,power_k.size):
                    for jk in xrange(0,power_k.size):
                        biased_cov[ik,jk]=pow(bias,4.)*power_pcov[ik,jk]+pow(bias,2.)*(power_pmean[ik]+power_pmean[jk])/ng+1./(ng*ng)
            fltformat="%-.12e"
            fout = open(covfile,"w")
            for i in xrange(0, power_k.size):
                for j in xrange(0, power_k.size):
                    fout.write(str(fltformat%biased_cov[i,j])+" ")
                fout.write("\n")
            fout.close()

    dalpha = 0.05
    dbeta = 0.05
    fisher = np.zeros((len(list_par),len(list_par)))
    npar=len(list_par)
    for ia in xrange(0,npar):
        ialpha=list_par[ia]
        for ib in xrange(0,npar):
            ibeta=list_par[ib]
            for i in xrange(0,len(list_aexp)):
                ioutput=list_noutput[i]
                aexp=list_aexp[i]

                dummy,Ppda=pkann_power(ialpha,dalpha,1,powertype,ioutput,mainpath,aexp,growth_a,growth_dplus)
                dummy,Pmda=pkann_power(ialpha,dalpha,-1,powertype,ioutput,mainpath,aexp,growth_a,growth_dplus)
                dummy,Pp2da=pkann_power(ialpha,dalpha,2,powertype,ioutput,mainpath,aexp,growth_a,growth_dplus)
                dummy,Pm2da=pkann_power(ialpha,dalpha,-2,powertype,ioutput,mainpath,aexp,growth_a,growth_dplus)
                dummy,Ppdb=pkann_power(ibeta,dbeta,1,powertype,ioutput,mainpath,aexp,growth_a,growth_dplus)
                dummy,Pmdb=pkann_power(ibeta,dbeta,-1,powertype,ioutput,mainpath,aexp,growth_a,growth_dplus)
                dummy,Pp2db=pkann_power(ibeta,dbeta,2,powertype,ioutput,mainpath,aexp,growth_a,growth_dplus)
                dummy,Pm2db=pkann_power(ibeta,dbeta,-2,powertype,ioutput,mainpath,aexp,growth_a,growth_dplus)

                deralpha=2.*(Ppda-Pmda)/(3.*dalpha)+(Pp2da-Pm2da)/(12.*dalpha)
                derbeta=2.*(Ppdb-Pmdb)/(3.*dbeta)+(Pm2db-Pp2db)/(12.*dbeta)
                
                inv_cov = linalg.inv(biased_cov)
                if (nsim>power_k.size+2):
                    inv_cov = float(nsim-power_k.size-2)*inv_cov/float(nsim-1)
                else:
                    print "warning: biased inverse covariance estimator"

                for ik in xrange(0,power_k.size):
                    for jk in xrange(0,power_k.size):
                        fisher[ia,ib]+=deralpha[ik]*derbeta[jk]*inv_cov[ik,jk]

    return fisher


def galaxy_bias(power_k=np.zeros(0), noutput=1):
    bias=np.zeros(power_k.size)
    for ik in xrange(0,power_k.size):
        bias[ik]=1.5

    return bias


def galaxy_density():
    ng=5.e-4
    return ng
