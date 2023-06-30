import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
from astropy import units as u
from astropy import constants as c
from glob import glob
import numpy.random as rand
from astropy.table import Table
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from utils import make_M1450_interp, gamma_qso
from m912 import m912

from IPython import embed

run_full_loop = True

z = 3.3 # For NB527
filt = "J"

gamma_UVB = 1.19275638e-12 * u.s ** -1  # Gamma_UVB at z = 3.3 (From 3.5 Nyx sims)
ang = 2.2 * u.arcmin  # Using sightline resolution from Koki NASA/Keck 2023A proposal

mean_sep = (1 + z) * Planck18.angular_diameter_distance(z) * ang.to(
    u.rad).value            # Mean sightline separation in pMpc

if run_full_loop:

    interp = make_M1450_interp() # Generate a M1450 -> Jmag interpolation
    
    M1450_range = np.arange(-22,-27,-0.01)  # Probe a range of M1450 values

    dist_vals = np.zeros_like(M1450_range)


    for i, M1450 in enumerate(M1450_range):

        Jmag  = np.float64(interp((M1450,z)))  # Obtain Jmag values to loop over

        logLv = m912(z, Jmag, filt)[1]

        gamma_QSO = gamma_qso(1 * (1 + z) * u.Mpc, z, logLv)[0]  # Find gamma_qso at 1 pMpc

        ratio = gamma_QSO / gamma_UVB  # Used to find required drop in gamma_qso due to distance

        equal_dist = np.sqrt(ratio[0].value) * (1 + z) * u.Mpc  # Distance at which gamma_qso = gamma_uvb in pMpc

        dist_vals[i] = equal_dist.value # Store equal_dist value for full comparison

    plt.plot(M1450_range, dist_vals, 'r', label='Distance to Gamma_QSO = Gamma_UVB in NB527 (z=3.3)')
    plt.hlines(mean_sep.value,-22,-27, 'r', linestyles='--', label='Mean Sightline Separation in NB527 (z=3.3)')
    plt.xlim(-22, -27)
    plt.xlabel(r'M_{1450}')
    plt.ylabel('Distance (pMpc)')
    plt.legend()
    plt.show()

    # np.savez('NB527_mag_dist', M1450=M1450_range, DIST=dist_vals)

else:  # Testing with M1450 = -25

    Jmag = 20.07096171  # Obtained manually from same interpolation as loop

    m912, logLv, logQ, m1216, logL1216, logFnu_norm = m912(z, Jmag, filt)  # Obtain logLv from Jmag value

    gamma_QSO = gamma_qso(1 * (1 + z) * u.Mpc, z, logLv)[0]  # Find gamma_qso at 1 pMpc

    ratio = gamma_QSO / gamma_UVB  # Used to find required drop in gamma_qso due to distance

    equal_dist = np.sqrt(ratio[0].value) * (1 + z) * u.Mpc  # Distance at which gamma_qso = gamma_uvb in pMpc

    print('\n')
    print("The distance until Gamma_QSO = Gamma_UVB with M1450 = -25:   " + str(np.round(equal_dist,4)) + '\n')
    print("The mean sightline separation at z = 3.3 in this field:      " + str(np.round(mean_sep, 4)))
    print('\n')

