import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebval as cval
from astropy.cosmology import Planck18
from astropy import units as u
from astropy import constants as c
from scipy.integrate import quad
from IPython import embed





#AGN Luminosity Function from Kulkarni Paper for Model 1
def phi_6(M):

    phi_star = 0.57 # 10^-3 Mpc^-3
    M_star = -20.87
    alpha = -1.87
    delta = 0.05

    term1 = 10**(-0.4*(M-M_star)*(alpha+1))
    term2 = -10**(-0.4*(M-M_star))

    return 1e-3*phi_star*(np.log(10)/2.5)*term1*np.exp(term2)

def app_to_abs_mag(m,z):

    return m + 2.5*np.log10(1+z) - 5*np.log10((Planck18.luminosity_distance(z).to(u.pc)/10).value)

m_low = 26
m_high = 24

#Volume of survey
z_816 = (8157.9/1215.7) - 1 #Using value of highest transmittance from HSC Filter files
z_816_low = (8113.4/1215.7) - 1 #Using lower value of 0.5 transmittance from HSC Filter Files
z_816_high = (8224.7/1215.7) - 1 #Using lower value of 0.5 transmittance from HSC Filter Files

z_872 = (8730.7/1215.7) - 1
z_872_low = (8704/1215.7) - 1
z_872_high = (8771/1215.7) - 1


FoV = 1.8 #deg^2; HSC FoV

ang_dist_816 = Planck18.angular_diameter_distance(z_816)
length_816 = Planck18.comoving_distance(z_816_high)-Planck18.comoving_distance(z_816_low)
vol_comov_816 = FoV * ((np.pi/180)*ang_dist_816)**2 * length_816
vol_816 = vol_comov_816 * (1+z_816)**3
M_816_low = app_to_abs_mag(m_low,z_816)

ang_dist_872 = Planck18.angular_diameter_distance(z_872)
length_872 = Planck18.comoving_distance(z_872_high)-Planck18.comoving_distance(z_872_low)
vol_comov_872 = FoV * ((np.pi/180)*ang_dist_872)**2 * length_872
vol_872 = vol_comov_872 * (1+z_872)**3
M_872_low = app_to_abs_mag(m_low,z_872)



lum_func_816 = phi_6(M_816_low) #Evaluate luminosity function at m_uv = 26
gal_count_816 = vol_816*lum_func_816*u.Mpc**-3 #Gal count per mpc^3 per mag

lum_func_872 = phi_6(M_872_low)
gal_count_872 = vol_816*lum_func_872*u.Mpc**-3


print("In NB816 (z=5.7) we predict: " + str(round(gal_count_816.value,1)) + " bg gals per mag for muv=26")
print("In NB872 (z=6.2) we predict: " + str(round(gal_count_872.value,1)) + " bg gals per mag for muv=26")


m_faint_vals = np.arange(18,32,0.01)
tot_count_816 = np.zeros_like(m_faint_vals)
tot_count_872 = np.zeros_like(m_faint_vals)

for i, m_faint in enumerate(m_faint_vals):
    lum_func_816 = quad(phi_6, -100, app_to_abs_mag(m_faint,z_816))[0]
    lum_func_872 = quad(phi_6, -100, app_to_abs_mag(m_faint,z_872))[0]

    tot_count_816[i] = vol_816.value * lum_func_816
    tot_count_872[i] = vol_872.value * lum_func_872


plt.plot(m_faint_vals,tot_count_816,label='NB816: Total bg Gals')
plt.plot(m_faint_vals,tot_count_872,label='NB872: Total bg Gals')
plt.yscale("log")
plt.xlim(18,32)
plt.ylim(1e-1,1e3)
plt.ylabel('bg Gal Count')
plt.xlabel('m_uv')
plt.grid()
plt.legend()
plt.show()



embed()