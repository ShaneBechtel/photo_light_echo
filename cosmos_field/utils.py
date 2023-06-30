import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits
from astropy import constants as c
from astropy import units as u
from astropy.cosmology import Planck18
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
from IPython import embed


def qso_template_nu(ALPHA_EUV=1.7):
    import numpy as np
    import astropy.units as u
    from astropy import constants as const
    import math
    import matplotlib.pyplot as plt
    from astropy.table import Table, QTable
    from astropy import constants as const
    from astropy.io import fits
    # from xastropy.xutils import xdebug as xdb
    from astropy.table import Table
    import os

    clight = const.c.to('cm/s').value
    angstrom = u.AA.to('cm')

    # Define template path
    templ_path = '/home/sbechtel/Documents/software/enigma/enigma/data/templates/'
    # Beta spliced to vanden Berk template with host galaxy  removed
    van_file = templ_path + 'VanDmeetBeta_fullResolution.txt'
    #van_file = '../data/templates/VanDmeetBeta_fullResolution.txt'
    nu_van, fnu_van = np.loadtxt(van_file, skiprows=1, unpack=True)
    isort = np.argsort(nu_van)
    nu_van = nu_van[isort]
    fnu_van = fnu_van[isort] / 1e-17
    lam_van = clight / nu_van / angstrom
    lognu_van = np.log10(nu_van)
    logfnu_van = np.log10(fnu_van)
    # Beta spliced to vanden Berk template with host galaxy removed
    gtr_file = templ_path + 'richards_2006_sed.txt'
    #gtr_file = '../data/templates/richards_2006_sed.txt'
    gtr = Table.read(gtr_file, format='ascii.cds')
    lognu_gtr = np.array(gtr['LogF'])
    logL_gtr = np.array(gtr['All'])
    isort = np.argsort(lognu_gtr)
    lognu_gtr = lognu_gtr[isort]
    logL_gtr = logL_gtr[isort]
    logLnu_gtr = logL_gtr - lognu_gtr  # Lnu in units of erg/s/Hz
    # create a vector of frequencies in log units
    lognu_min = lognu_gtr.min()
    # This is about 1e-3 Rydberg, limit of richards template
    lognu_max = math.log10((1e5 * c.Ryd).to('Hz',
                                                equivalencies=u.spectral()).value)
    # This is 1.36 Mev
    dlognu = 0.0001  # SDSS pixel scale
    lognu = np.arange(lognu_min, lognu_max, dlognu)
    nu = np.power(lognu, 10)
    n_nu = len(nu)
    logfnu = np.zeros(n_nu)
    # Some relevant frequencies
    lognu_2500 = math.log10((2500.0 * u.AA).to('Hz',  # 2500A for alpha_OX
                                               equivalencies=u.spectral()).value)
    lognu_8000 = math.log10((8000.0 * u.AA).to('Hz',  # IR splice is at 8000A
                                               equivalencies=u.spectral()).value)
    lognu_1Ryd = math.log10(u.rydberg.to('Hz',
                                         equivalencies=u.spectral()))
    lognu_30Ryd = math.log10((30.0 * u.rydberg).to('Hz',
                                                   equivalencies=u.spectral()).value)
    lognu_2keV = math.log10((2.0 * u.keV).to('Hz',
                                             equivalencies=u.spectral()).value)
    lognu_100keV = math.log10((100.0 * u.keV).to('Hz',
                                                 equivalencies=u.spectral()).value)

    # compute median as template is noisy here
    logfnu_van_8000 = np.median(logfnu_van[((lam_van >= 8000) & (lam_van <= 8100))])
    logLnu_gtr_8000 = np.interp(lognu_8000, lognu_gtr, logLnu_gtr)
    logLnu_gtr_2500 = np.interp(lognu_2500, lognu_gtr, logLnu_gtr)
    # IR part: nu < 8000A/c;  use the Richards al. 2006 template
    i_8000 = np.where(lognu <= lognu_8000)
    logfnu[i_8000] = np.interp(lognu[i_8000], lognu_gtr, logLnu_gtr)
    # UV part: c/8000A < nu < 1 Ryd/h; use the template itself
    i_UV = np.where((lognu > lognu_8000) & (lognu <= lognu_1Ryd))
    logfnu[i_UV] = (logLnu_gtr_8000 - logfnu_van_8000) + np.interp(lognu[i_UV], lognu_van, logfnu_van)
    logfnu_van_1Ryd = (logLnu_gtr_8000 -
                       logfnu_van_8000) + np.interp(lognu_1Ryd, lognu_van, logfnu_van)
    logfnu_2500 = np.interp(lognu_2500, lognu, logfnu)
    # This equation is from Strateva et al. 2005 alpha_OX paper.  I'm
    # evaluating the alpha_OX at the L_nu of the template, which is
    # based on the normalization of the Richards template. A more
    # careful treatment would actually use the 2500A luminosity of the
    # quasar template, after it is appropriately normalized
    alpha_OX = -0.136 * logfnu_2500 + 2.630
    logfnu_2keV = logfnu_2500 + alpha_OX * (lognu_2keV - lognu_2500)
    # FUV par 1 Ryd/h < nu < 30 Ryd/h, use the alpha_EUV power law
    i_FUV = np.where((lognu > lognu_1Ryd) & (lognu <= lognu_30Ryd))
    logfnu[i_FUV] = logfnu_van_1Ryd - ALPHA_EUV * (lognu[i_FUV] - lognu_1Ryd)
    logfnu_30Ryd = logfnu_van_1Ryd - ALPHA_EUV * (lognu_30Ryd - lognu_1Ryd)
    # soft X-ray part 30 Ryd/h < nu < 2kev/h;
    # use a power law with a slope alpha_soft chosen to match the fnu_2Kev implied by the alpha_OX
    i_soft = np.where((lognu > lognu_30Ryd) & (lognu <= lognu_2keV))
    alpha_soft = (logfnu_2keV - logfnu_30Ryd) / (lognu_2keV - lognu_30Ryd)
    logfnu[i_soft] = logfnu_30Ryd + alpha_soft * (lognu[i_soft] - lognu_30Ryd)
    # X-ray part 2 kev/h < nu < 100 keV/h
    i_X = np.where((lognu > lognu_2keV) & (lognu <= lognu_100keV))
    alpha_X = -1.0  # adopt this canonical 'flat X-ray slope'
    logfnu[i_X] = logfnu_2keV + alpha_X * (lognu[i_X] - lognu_2keV)
    logfnu_100kev = logfnu_2keV + alpha_X * (lognu_100keV - lognu_2keV)
    # hard X-ray part nu > 100 keV/h
    i_HX = np.where(lognu > lognu_100keV)
    alpha_HX = -2.0  # adopt this canonical 'flat X-ray slope'
    logfnu[i_HX] = logfnu_100kev + alpha_HX * (lognu[i_HX] - lognu_100keV)

    return lognu, logfnu


def sdss_filter_nu(lognu):
    import numpy as np
    import astropy.units as u
    from astropy import constants as c
    import math
    import matplotlib.pyplot as plt
    from astropy.table import Table, QTable
    from astropy import constants as c
    from astropy.io import fits
    # from xastropy.xutils import xdebug as xdb
    from scipy.integrate import simps
    from scipy.interpolate import interp1d
    from scipy.integrate import simps
    import os

    clight = c.c.to('cm/s').value
    angstrom = u.AA.to('cm')

    # Files for filter curves
    # Define filter path
    filt_path = '/home/sbechtel/Documents/software/enigma/enigma/data/filter_curves/'

    u_file = filt_path + 'sdss_u0.res'
    g_file = filt_path + 'sdss_g0.res'
    r_file = filt_path + 'sdss_r0.res'
    i_file = filt_path + 'sdss_i0.res'
    z_file = filt_path + 'sdss_z0.res'
    B_file = filt_path + 'john_b.dat'
    BJ_file = filt_path + 'bj.dat'
    J_file = filt_path + 'niri_J.dat'
    H_file = filt_path + 'niri_H.dat'
    K_file = filt_path + 'niri_K.dat'

    lam_u, F_u13 = np.loadtxt(u_file, unpack=True, usecols=(0, 3), comments='\ ')
    lam_g, F_g13 = np.loadtxt(g_file, unpack=True, usecols=(0, 3), comments='\ ')
    lam_r, F_r13 = np.loadtxt(r_file, unpack=True, usecols=(0, 3), comments='\ ')
    lam_i, F_i13 = np.loadtxt(i_file, unpack=True, usecols=(0, 3), comments='\ ')
    lam_z, F_z13 = np.loadtxt(z_file, unpack=True, usecols=(0, 3), comments='\ ')

    # These are energy efficiencies, so need to be divided by lam to be
    # canonical quantum efficiencies. The normalization is such that
    # HD19445 has B-V of 0.46 B filter is at 1 airmass whereas sdss is at 1.3???
    lam_B, F_B1_temp = np.loadtxt(B_file, unpack=True,
                                  usecols=(0, 2), comments='\ ')
    F_B1 = F_B1_temp / lam_B
    # UKSTU B_J - 2mm GG395 with UJ10738P transmission,
    # UKST optics and unit airmass
    lam_B_J, F_BJ1 = np.loadtxt(BJ_file, unpack=True, usecols=(0, 1), comments='#')
    # These are warm NIRI IR filters take from the NIRI webpage
    # http://www.gemini.edu/sciops/instruments/niri/NIRIFilterList.html
    lam_J, F_J1 = np.loadtxt(J_file, unpack=True, usecols=(0, 1), comments='#')
    lam_H, F_H1 = np.loadtxt(H_file, unpack=True, usecols=(0, 1), comments='#')
    lam_K, F_K1 = np.loadtxt(K_file, unpack=True, usecols=(0, 1), comments='#')

    lognu_u = np.log10(((lam_u[::-1] * u.AA).to('Hz', equivalencies=u.spectral())).value)
    lognu_g = np.log10(((lam_g[::-1] * u.AA).to('Hz', equivalencies=u.spectral())).value)
    lognu_r = np.log10(((lam_r[::-1] * u.AA).to('Hz', equivalencies=u.spectral())).value)
    lognu_i = np.log10(((lam_i[::-1] * u.AA).to('Hz', equivalencies=u.spectral())).value)
    lognu_z = np.log10(((lam_z[::-1] * u.AA).to('Hz', equivalencies=u.spectral())).value)
    lognu_B = np.log10(((lam_B[::-1] * u.AA).to('Hz', equivalencies=u.spectral())).value)
    lognu_B_J = np.log10(((lam_B_J[::-1] * u.AA).to('Hz',
                                                    equivalencies=u.spectral())).value)
    # These were in descending order so no reverse
    lognu_J = np.log10(((lam_J * u.nm).to('Hz', equivalencies=u.spectral())).value)
    lognu_H = np.log10(((lam_H * u.nm).to('Hz', equivalencies=u.spectral())).value)
    lognu_K = np.log10(((lam_K * u.nm).to('Hz', equivalencies=u.spectral())).value)
    # filter transmissions in aribtrary units
    F_u = np.power(10.0, 2.0 * lognu_u - 30.0) * F_u13[::-1]
    F_g = np.power(10.0, 2.0 * lognu_g - 30.0) * F_g13[::-1]
    F_r = np.power(10.0, 2.0 * lognu_r - 30.0) * F_r13[::-1]
    F_i = np.power(10.0, 2.0 * lognu_i - 30.0) * F_i13[::-1]
    F_z = np.power(10.0, 2.0 * lognu_z - 30.0) * F_z13[::-1]
    F_B = np.power(10.0, 2.0 * lognu_B - 30.0) * F_B1[::-1]
    F_B_J = np.power(10.0, 2.0 * lognu_B_J - 30.0) * F_BJ1[::-1]
    # Don't reverse these
    F_J = np.power(10.0, 2.0 * lognu_J - 30.0) * F_J1
    F_H = np.power(10.0, 2.0 * lognu_H - 30.0) * F_H1
    F_K = np.power(10.0, 2.0 * lognu_K - 30.0) * F_K1

    n_nu = lognu.size

    filt = Table([np.zeros(n_nu), np.zeros(n_nu), np.zeros(n_nu), np.zeros(n_nu),
                  np.zeros(n_nu), np.zeros(n_nu), np.zeros(n_nu), np.zeros(n_nu),
                  np.zeros(n_nu), np.zeros(n_nu)],
                 names=('u', 'g', 'r', 'i', 'z', 'B', 'B_J', 'J', 'H', 'K'))

    ind_u, = np.where((lognu >= lognu_u.min()) & (lognu <= lognu_u.max()))
    ind_g,  = np.where((lognu >= lognu_g.min()) & (lognu <= lognu_g.max()))
    ind_r, = np.where((lognu >= lognu_r.min()) & (lognu <= lognu_r.max()))
    ind_i, = np.where((lognu >= lognu_i.min()) & (lognu <= lognu_i.max()))
    ind_z, = np.where((lognu >= lognu_z.min()) & (lognu <= lognu_z.max()))
    ind_B, = np.where((lognu >= lognu_B.min()) & (lognu <= lognu_B.max()))
    ind_B_J, = np.where((lognu >= lognu_B_J.min()) & (lognu <= lognu_B_J.max()))
    ind_J, = np.where((lognu >= lognu_J.min()) & (lognu <= lognu_J.max()))
    ind_H, = np.where((lognu >= lognu_H.min()) & (lognu <= lognu_H.max()))
    ind_K, = np.where((lognu >= lognu_K.min()) & (lognu <= lognu_K.max()))

    spl_u = interp1d(lognu_u, F_u)
    spl_g = interp1d(lognu_g, F_g)
    spl_r = interp1d(lognu_r, F_r)
    spl_i = interp1d(lognu_i, F_i)
    spl_z = interp1d(lognu_z, F_z)
    spl_B = interp1d(lognu_B, F_B)
    spl_B_J = interp1d(lognu_B_J, F_B_J)
    spl_J = interp1d(lognu_J, F_J)
    spl_H = interp1d(lognu_H, F_H)
    spl_K = interp1d(lognu_K, F_K)

    F_u0 = spl_u(lognu[ind_u])
    F_g0 = spl_g(lognu[ind_g])
    F_r0 = spl_r(lognu[ind_r])
    F_i0 = spl_i(lognu[ind_i])
    F_z0 = spl_z(lognu[ind_z])
    F_B0 = spl_B(lognu[ind_B])
    F_B_J0 = spl_B_J(lognu[ind_B_J])
    F_J0 = spl_J(lognu[ind_J])
    F_H0 = spl_H(lognu[ind_H])
    F_K0 = spl_K(lognu[ind_K])

    # Normalize filter curves to unity in integral against lognu
    filt['u'][ind_u] = F_u0
    filt['g'][ind_g] = F_g0
    filt['r'][ind_r] = F_r0
    filt['i'][ind_i] = F_i0
    filt['z'][ind_z] = F_z0
    filt['B'][ind_B] = F_B0
    filt['B_J'][ind_B_J] = F_B_J0
    filt['J'][ind_J] = F_J0
    filt['H'][ind_H] = F_H0
    filt['K'][ind_K] = F_K0

    LN10 = math.log(10.0)
    for col in filt.colnames:
        norm = simps(LN10 * filt[col], lognu)
        filt[col] = filt[col] / norm

    return filt


def lamL_lam(znow, m, filt, lam, cosmo=False, ALPHA_EUV=1.7, MASKLYA=False, CLOUDY=False,
         IGNORE=False):
    import astropy.units as u
    from astropy import constants as c
    import math
    import matplotlib.pyplot as plt
    from astropy import constants as c
    from astropy.io import fits
    #from xastropy.xutils import xdebug as xdb
    # from qso_template_nu import qso_template_nu
    # from sdss_filter_nu import sdss_filter_nu
    from scipy.integrate import simps

    znow = np.asarray([znow]) if (np.isscalar(znow)) else np.asarray(znow)
    m = np.asarray([m]) if (np.isscalar(m)) else np.asarray(m)

    # Error checkin on sizes of inputs
    if (len(znow) != len(m)):
        raise Exception('The number of elements in znow and m must be the same')

    # If filter has length 1, then replicate it to the size of m and z
    if (len(filt) == len(znow)):
        filt = tuple(filt)
    elif ((len(znow) > 1) & (len(filt) == 1)):
        filt = len(znow) * filt

    if (cosmo == False):
        from astropy.cosmology import Planck15 as cosmo

    ## CLOUDY and LOGLNU functionality of the original IDL routine not yet
    ## supported
    if (CLOUDY == True):
        if (znow.size != 1 | m.size != 1):
            raise Exception('CLOUDY options only supported for scalar inputs')

    LN10 = math.log(10.0)
    HORIZON = 2.9979246e9  # HORIZON in pc/h
    angstrom = u.AA.to('cm')
    c = c.c.to('cm/s').value
    nu_lam = (lam*u.AA).to('Hz', equivalencies=u.spectral()).value
    lognu_lam = np.log10(nu_lam)

    AB_source = 3631.0
    # The standard AB magnitude reference flux is 3631 Jy = 3631*1.0e-23 erg/cm^2/s/Hz

    # effective beginning wavelengths of SDSS filters. Defined to be the wavelength wehre filters are 10% of their
    # peak values
    filter_names = {'u': 3125.0, 'g': 3880.0, 'r': 5480.0, 'i': 6790.0,
                    'z': 8090.0, 'B': 3772.0, 'B_J': 3706.3}
    filt_wave = np.zeros(len(filt))
    for ii in range(len(filt)):
        try:
            filt_wave[ii] = filter_names.get(filt[ii])
        except:
            print('Undientified filters in filt')

    # Identify bad indices where Ly-a is within the filter
    bad_inds, = np.where((1.0 + znow) * 1215.67 > filt_wave)
    # xdb.set_trace()
    if len(bad_inds) != 0 and not IGNORE:
        raise Exception('The filter begins at lam=' + str(filt_wave[bad_inds]) +
                        ' which must be redder than the observed frame Lya at lam=' +
                        str((1.0 + znow[bad_inds]) * 1215.67) +
                        ' . There are ' + str(len(bad_inds)) + ' bad redshifts')

    nz = znow.size
    m_lam = np.zeros(nz)
    L_lam = np.zeros(nz)
    logFnu_norm = np.zeros(nz)
    A_norm = np.zeros(nz)
    anow = 1.0 / (1.0 + znow)

    # Read in the composite quasar template
    lognu, logFnu_beta = qso_template_nu(ALPHA_EUV=ALPHA_EUV)
    logFnu_beta = logFnu_beta
    # evalute template at lambda
    logFnu_beta_lam = np.interp(lognu_lam, lognu, logFnu_beta)

    # compute filter curve table at these lognu
    filt_table = sdss_filter_nu(lognu)

    for iz in range(nz):
        # redshift the rest frame spectrum to the observed frame
        # and integrate the redshifted spectrum over the filter curve
        # xdb.set_trace()
        lognu_rest = lognu + math.log10(1.0 + znow[iz])
        logFnu_beta_rest = np.interp(lognu_rest, lognu, logFnu_beta)
        Fnu_rest = np.power(10.0, logFnu_beta_rest)
        F_m = filt_table[filt[iz]]
        obs = LN10 * Fnu_rest * F_m
        obs_integ = simps(obs, lognu)
        m_lam[iz] = -2.5 * (logFnu_beta_lam - math.log10(obs_integ)) + m[iz]
        logFnu_norm[iz] = -math.log10(obs_integ) - 0.4 * (m[iz] + 48.6)

    DL1 = (cosmo.luminosity_distance(znow)).to('cm')
    logDL = np.log10(DL1.value)
    logLv = np.log10(anow * 4.0 * math.pi * AB_source) + 2.0 * logDL - 0.4 * m_lam - 23.0
    lam_L_lam = nu_lam*np.power(10.0,logLv)

    ## To be added: CLOUDYFILE optoins, and option to pass back the LOGLNU
    # LOGLNU = logLv[0]  + logFnu_beta - logFnu_beta912

    # xdb.set_trace()
    return lam_L_lam





def Jmag_to_M1450(Jmag, z, cosmology=Planck18, ALPHA_EUV=1.7):
    lam_l = lamL_lam(z, Jmag, 'J', 1450.0, cosmo=cosmology, IGNORE=False, ALPHA_EUV=ALPHA_EUV) * u.erg / u.s
    lognu_1450 = np.log10((1450.0 * u.Angstrom).to('Hz', equivalencies=u.spectral()).value)

    Lnu_1450 = lam_l / (np.power(10.0, lognu_1450) * u.Hz)
    fnu_1450 = (Lnu_1450 / 4.0 / np.pi / (10.0 * u.pc) ** 2).to('Jansky')
    Mnu_1450 = -2.5 * np.log10(fnu_1450 / (3631.0 * u.Jansky))

    return Mnu_1450



def get_interp_2_to_1(f, x1_range, x2_range, ngrid_x1=51, ngrid_x2=51, mode='forward'):
    """
    Routine to create an interpolator for a function f: (x1, x2) -> y or its inverse w.r.t. the first argument.

    Args:
        f (function):
            original function f: (x1, x2) -> y
        x1_range (tuple):
            range for the first argument of f
        x2_range (tuple):
            range for the second argument of f
        ngrid_x1 (int):
            number of interpolation grid points for the first argument of f
        ngrid_x2 (int):
            number of interpolation grid points for the second argument of f
        mode (str):
            if mode == 'forward':
                create a regular grid interpolator for f: (x1, x2) -> y
            elif mode == 'inverse':
                create an interpolator for f_inv: (y, x2) -> x1
    Returns:
        interp (function):
            interpolator
    """

    x1_grid = np.linspace(*x1_range, ngrid_x1)
    x2_grid = np.linspace(*x2_range, ngrid_x2)
    x2_GRID, x1_GRID = np.meshgrid(x2_grid, x1_grid)
    y_GRID = np.zeros((ngrid_x1, ngrid_x2))
    for i in range(ngrid_x1):
        for j in range(ngrid_x2):
            y_GRID[i,j] = f(x1_grid[i], x2_grid[j])
    if mode == 'forward':
        interp = lambda x: RegularGridInterpolator((x1_grid, x2_grid), y_GRID)(x)
    elif mode == 'inverse':
        interp = lambda x: LinearNDInterpolator(list(zip(y_GRID.flatten(), x2_GRID.flatten())), x1_GRID.flatten())(x)
    return interp


def make_M1450_interp():
    Jmag_min = 15
    Jmag_max = 25
    z_min = 0.1
    z_max = 7

    M1450_to_Jmag_interp = get_interp_2_to_1(Jmag_to_M1450, (Jmag_min, Jmag_max), (z_min, z_max), ngrid_x1=51,
                                             ngrid_x2=51, mode='inverse')

    return M1450_to_Jmag_interp

def gamma_qso(R, z, logLv, ALPHA_EUV=1.7):
    """

    Args:
        R: Distance in comoving Mpc at which the \Gamma_QSO is desired
        z: Redshift in question
        logLv: Specific luminosity at the Lyman edge
        ALPHA_EUV: EUV Spectral slope to assume

    Returns:

    """

    aa = 1.0 / (1.0 + z)
    sigma_LL = 6.30e-18 * u.cm**2
    L_nu = np.power(10.0, logLv) * u.erg / u.s / u.Hz
    gamma = sigma_LL * L_nu / 4.0 / np.pi / c.h / (ALPHA_EUV + 3.0) / (aa * R)**2
    gamma_val = (gamma.decompose()).cgs  # .value/1e-12
    HNUBAR = (u.rydberg / (ALPHA_EUV + 2.0)).cgs

    return gamma_val, HNUBAR
