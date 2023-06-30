#+
# NAME:
#   m912
#
# PURPOSE:
#   Compute the apparent magnitude one would measure at the 
#   Lyman limit in the observed frame at lambda = (1 + z)912
#
# CALLING SEQUENCE:
#   m_912 = m912(z,m,filter,omega_m,omega_v,w,lit_h [ logLv = ,
#           ALPHA_EUV = , IGNORE ]) 

# INPUTS:
#   z         -  Redshift of the quasar
#   m         -  Apparent magnitude
#   filter    -  filter name SDSS [u,g,r,i,z,B,B_J]
#   omega_m   - Matter density
#   omega_v   - Dark energy density 
#   W         - Equation of state parameter.
#   lit_h     - Dimensionless Hubble constant
#
# OPTIONAL KEYWORDS:
#   logLv      - Log_10 of the specific luminosity at the Lyman edge 
#                (ergs/s/Hz)
#   ALPHA_EUV  - Spectral slope in the extreme UV. Default is 1.57   
#                (Tefler et al 2002). A power law v^{-ALPHA_EUV} 
#                is pasted onto the quasar spectrum at 1285 A. 

# OUTPUTS:
#   m_912       - Apparent magnitude one would measure at the Lyman 
#                 limit in the observed frame at lambda = (1 + z)912
#
# COMMENTS:
#   The code will return an error if one tries to use a filter with 
#   lambda_min < (1 + z)*1216. The lyman alpha forest will reduce the 
#   observed apparent magnitude, and the mean flux blueward of 1216 
#   is not reproduced correctly in the  quasar template used here. 
# 
# EXAMPLE:
#     Compute m_912 for a quasar at z=2.43 which has g=20.5
#     m_912=m912(2.43,20.5,1,logLv=logLv)
#     print,m_912
#       22.199551
#     print,loglv
#       29.812608
#
# PROCEDURES CALLED:
#   dofa()
#   sdss_filter_nu()
#   qso_template_nu()
#   obs_int_nu()
#   
# REVISION HISTORY:
#   17-Sep-2005  Written by Joe Hennawi UCB 
#   12-Aug-2015  ported to python
#------------------------------------------------------------------------------

import numpy as np
from astropy.table import Table, QTable
from IPython import embed


def lamL_lam(znow, m, filt, lam, cosmo=False, ALPHA_EUV=1.7, MASKLYA=False, CLOUDY=False,
         IGNORE=False):
    import astropy.units as u
    from astropy import constants as const
    import math
    import matplotlib.pyplot as plt
    from astropy import constants as const
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
    c = const.c.to('cm/s').value
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


def m912(znow,m,filt,cosmo=False,ALPHA_EUV=1.7,MASKLYA=False,CLOUDY=False,
         IGNORE=False):

    import astropy.units as u
    from astropy import constants as const
    import math
    import matplotlib.pyplot as plt
    from astropy import constants as const
    from astropy.io import fits
    #from xastropy.xutils import xdebug as xdb
    #from qso_template_nu import qso_template_nu
    #from sdss_filter_nu import sdss_filter_nu
    from scipy.integrate import simps
    #from xastropy.xutils import xdebug as xdb

    znow= np.asarray([znow]) if (np.isscalar(znow)) else np.asarray(znow)
    m= np.asarray([m]) if (np.isscalar(m)) else np.asarray(m)

    # Error checkin on sizes of inputs
    if (len(znow) != len(m)):
        raise Exception('The number of elements in znow and m must be the same')

    # If filter has length 1, then replicate it to the size of m and z
    if (len(filt) == len(znow)):
        filt=tuple(filt)
    elif ((len(znow) > 1) & (len(filt) == 1)):
        filt = len(znow)*filt

    if (cosmo==False):
        from astropy.cosmology import Planck15 as cosmo

    ## CLOUDY and LOGLNU functionality of the original IDL routine not yet
    ## supported
    if (CLOUDY==True):
        if (znow.size != 1 | m.size != 1):
            raise Exception('CLOUDY options only supported for scalar inputs')
    
    LN10=math.log(10.0)
    HORIZON = 2.9979246e9 # HORIZON in pc/h
    angstrom=u.AA.to('cm')
    c=const.c.to('cm/s').value
    lognu_912 = math.log10(u.rydberg.to('Hz',
                                         equivalencies=u.spectral()))
    nu_1216l=  (1210.0*u.AA).to('Hz',equivalencies=u.spectral()).value
    lognu_1216l=math.log10((1222.0*u.AA).to('Hz',equivalencies=u.spectral()).value)
    lognu_1216r=math.log10((1210.0*u.AA).to('Hz',equivalencies=u.spectral()).value)
    
    AB_source = 3631.0
    # The standard AB magnitude reference flux is 3631 Jy = 3631*1.0e-23 erg/cm^2/s/Hz

    # effective beginning wavelengths of SDSS filters. Defined to be the wavelength wehre filters are 10% of their
    # peak values
    filter_names = {'u': 3125.0, 'g': 3880.0, 'r': 5480.0, 'i': 6790.0,
                    'z': 8090.0, 'J': 11700.0,'H': 14900.0,'K': 22000.0,'B': 3772.0, 'B_J': 3706.3}
    filt_wave = np.zeros(len(filt))
    for ii in range(len(filt)):
        if(filter_names.get(filt[ii])):
            filt_wave[ii]=filter_names.get(filt[ii])
        else:
            raise Exception('Undientified filters in filt')
            
    # Identify bad indices where Ly-a is within the filter        
    bad_inds, = np.where((1.0 + znow)*1215.67 > filt_wave)
    #xdb.set_trace()
    if len(bad_inds) != 0 and not IGNORE:
        raise Exception('The filter begins at lam=' + str(filt_wave[bad_inds]) +
                        ' which must be redder than the observed frame Lya at lam=' +
                        str((1.0 + znow[bad_inds])*1215.67) +
                        ' . There are ' + str(len(bad_inds)) + ' bad redshifts')

    nz=znow.size
    m912=np.zeros(nz)
    m1216=np.zeros(nz)
    logFnu_norm=np.zeros(nz)
    A_norm=np.zeros(nz)
    anow=1.0/(1.0 + znow)

    # Read in the composite quasar template
    lognu,logFnu_beta = qso_template_nu(ALPHA_EUV=ALPHA_EUV)
    logFnu_beta = logFnu_beta
    # evalute template at 912 and 1216
    logFnu_beta912 = np.interp(lognu_912,lognu,logFnu_beta)

    lya_pix = np.where((lognu >= lognu_1216l) & (lognu <= lognu_1216r))
    logFnu_beta1216 = logFnu_beta[lya_pix].max()

    # compute filter curve table at these lognu
    filt_table=sdss_filter_nu(lognu)
    
    for iz in range(nz):
        # redshift the rest frame spectrum to the observed frame
        # and integrate the redshifted spectrum over the filter curve
        #xdb.set_trace()
        lognu_rest = lognu + math.log10(1.0 + znow[iz])
        logFnu_beta_rest = np.interp(lognu_rest,lognu,logFnu_beta)
        Fnu_rest=np.power(10.0,logFnu_beta_rest)
        F_m=filt_table[filt[iz]]
        obs = LN10*Fnu_rest*F_m
        obs_integ = simps(obs,lognu)
        m912[iz] = -2.5*(logFnu_beta912 - math.log10(obs_integ)) + m[iz]
        m1216[iz] = -2.5*(logFnu_beta1216 - math.log10(obs_integ)) + m[iz]
        logFnu_norm[iz]=-math.log10(obs_integ) - 0.4*(m[iz] + 48.6)


    DL1=(cosmo.luminosity_distance(znow)).to('cm') 
    logDL = np.log10(DL1.value)
    logLv = np.log10(anow*4.0*math.pi*AB_source) + 2.0*logDL - 0.4*m912 - 23.0
    logL1216 = np.log10(anow*4.0*math.pi*AB_source) + 2.0*logDL - 0.4*m1216 - 23.0
    logQ = logLv - math.log10(const.h.to('erg s').value) - math.log10(ALPHA_EUV)

    ## To be added: CLOUDYFILE optoins, and option to pass back the LOGLNU
    #LOGLNU = logLv[0]  + logFnu_beta - logFnu_beta912

    #xdb.set_trace()
    #print('LogLv: ' + str(logLv[0]))
    return (m912, logLv, logQ, m1216, logL1216, logFnu_norm)



# This is a wrapper for m912 to deal with objects which have 5-band SDSS photometry
def m912_mags(zin, mags1, cosmo=False, ALPHA_EUV = 1.7, IGNORE=False):

    znow = np.atleast_1d(zin)
    nz = len(znow)
    # Error checking on size of mags
    if nz==1:
        mags = np.reshape(mags1,(nz,5))
    else:
        mags = mags1
    if(mags.shape) != (nz,5):
        raise ValueError('Problem with the size of mags')

    # effective beginning wavelengths of SDSS filters. Defined to be the wavelength wehre filters are 10% of their
    # peak values
    filter_names = ['u','g','r','i','z']
    wave_beg = [3125.0,3880.0,5480.0,6790.0,8090.0]
    filt_tab = Table([filter_names, wave_beg], names = ('filt_name', 'wave_beg'))

    m_912 = np.zeros(nz)
    logLv = np.zeros(nz)

    for iz in range(nz):
        lam_ly = (1.0 + znow[iz])*1215.67
        good_inds, = np.where(filt_tab['wave_beg'] >= lam_ly)
        if len(good_inds) > 0:
            filter_ind = good_inds.min()
            filter_str = filt_tab['filt_name'][filter_ind]
            mag_now = mags[iz,filter_ind]
            m1, logL1, _, _, _, _ = m912(znow[iz],mag_now,filter_str,cosmo=cosmo, ALPHA_EUV = ALPHA_EUV, IGNORE=IGNORE)
            m_912[iz] = m1
            logLv[iz] = logL1

    return m_912, logLv


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
    lognu_max = math.log10((1e5 * const.Ryd).to('Hz',
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

    # i8000, = np.where(((lam_van >= 8000.0) and (lam_van <= 8100.0)))


    # 2500A for alpha_OX
    # lognu_8000 =   alog10(c.c/8000.d/angstrom) # IR splice is at 8000A
    # lognu_1Ryd = alog10(c.Ryd/c.h)
    # lognu_30Ryd = alog10(30.0d*c.Ryd/c.h)
    # lognu_2keV = alog10(c.ev*2000.0/c.h)
    # lognu_100keV = alog10(c.ev*1d5/c.h)


def sdss_filter_nu(lognu):
    import numpy as np
    import astropy.units as u
    from astropy import constants as const
    import math
    import matplotlib.pyplot as plt
    from astropy.table import Table, QTable
    from astropy import constants as const
    from astropy.io import fits
    # from xastropy.xutils import xdebug as xdb
    from scipy.integrate import simps
    from scipy.interpolate import interp1d
    from scipy.integrate import simps
    import os

    clight = const.c.to('cm/s').value
    angstrom = u.AA.to('cm')

    # Files for filter curves
    # Define filter path
    filt_path = '/home/sbechtel/Documents/software/enigma/enigma/data/filter_curves/'
    #filt_path = '../data/filter_curves/'

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
