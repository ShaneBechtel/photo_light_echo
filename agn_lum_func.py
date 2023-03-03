import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebval as cval
from astropy.cosmology import Planck18
from astropy import units as u
from astropy import constants as c
from scipy.integrate import quad
from IPython import embed

#Using Model 1 from Kulkarni Paper
def F0(z,c0):
    return cval(z,c0)

def F1(z,c1):
    return cval(z,c1)

def F2(z,c2):
    return cval(z,c2)

def F3(z,c3):
    xi = np.log10(z/(1+c3[2]))
    denom = 10**(c3[3]*xi) + 10**(c3[4]*xi)
    return c3[0] + c3[1]/denom

#List of coeffs from Kulkarni Paper for Model 1
c_val0 = [-7.798,1.128,-0.12]
c_val1 = [-17.163,-5.512,0.593,-0.024]
c_val2 = [-3.223,-0.258]
c_val3 = [-2.312,0.559,3.773,141.884,-0.171]

c_vals = [c_val0,c_val1,c_val2,c_val3]

#AGN Luminosity Function from Kulkarni Paper for Model 1
def phi(M,z,c_vals):

    phi_star = 10**F0(1+z,c_vals[0])
    M_star = F1(1+z,c_vals[1])
    alpha = F2(1+z,c_vals[2])
    beta = F3(1+z,c_vals[3])

    term1 = 10**(0.4*(1+alpha)*(M-M_star))
    term2 = 10**(0.4*(1+beta)*(M-M_star))

    denom = term1+term2

    return phi_star/denom #phi/cMpc**-3 * mag**-1

def phi_718(M):

    phi_star = 10**(-9.03)
    M_star = -27.89
    alpha = -4.55
    beta = -2.31

    term1 = 10**(0.4*(1+alpha)*(M-M_star))
    term2 = 10**(0.4*(1+beta)*(M-M_star))

    denom = term1+term2

    return phi_star/denom #phi/cMpc**-3 * mag**-1


#Volume of survey
z_527 = (5249.1/1215.7) - 1 #Using value of highest transmittance from HSC Filter files
z_527_low = (5215.5/1215.7) - 1 #Using lower value of 0.5 transmittance from HSC Filter Files
z_527_high = (5291.2/1215.7) - 1 #Using lower value of 0.5 transmittance from HSC Filter Files

z_718 = (7164.4/1215.7) - 1
z_718_low = (7111.9/1215.7) - 1
z_718_high = (7221.4/1215.7) - 1

z_816 = (8157.9/1215.7) - 1
z_816_low = (8113.4/1215.7) - 1
z_816_high = (8224.7/1215.7) - 1


ang_radius = 1.67/2 #degrees, area of Cosmos field

ang_dist_527 = Planck18.angular_diameter_distance(z_527)
length_527 = Planck18.comoving_distance(z_527_high)-Planck18.comoving_distance(z_527_low)
vol_comov_527 = np.pi * (ang_radius * (np.pi/180) * ang_dist_527)**2 * length_527
vol_527 = vol_comov_527 * (1+z_527)**3

ang_dist_718 = Planck18.angular_diameter_distance(z_718)
length_718 = Planck18.comoving_distance(z_718_high)-Planck18.comoving_distance(z_718_low)
vol_comov_718 = np.pi * (ang_radius * (np.pi/180) * ang_dist_718)**2 * length_718
vol_718 = vol_comov_718 * (1+z_718)**3

ang_dist_816 = Planck18.angular_diameter_distance(z_816)
length_816 = Planck18.comoving_distance(z_816_high)-Planck18.comoving_distance(z_816_low)
vol_comov_816 = np.pi * (ang_radius * (np.pi/180) * ang_dist_816)**2 * length_816
vol_816 = vol_comov_816 * (1+z_816)**3


#f=1-np.cos(np.pi/6) #Assume opening angle of 30 degs
f = 1 - 0.6 #0.6 = covering factor found in literature

#lower limit of -25 for mag
lum_func_527 = phi(-25,z_527,c_vals) #Evaluate luminosity function at -25 Mag
agn_count_obs_527 = vol_527*lum_func_527*u.Mpc**-3 #Obstructed AGN count
agn_count_527 = agn_count_obs_527/f #Unobstructed AGN count

lum_func_718 = phi(-25,z_718,c_vals)
agn_count_obs_718 = vol_718*lum_func_718*u.Mpc**-3
agn_count_718 = agn_count_obs_718/f

lum_func_816 = phi(-25,z_816,c_vals)
agn_count_obs_816 = vol_816*lum_func_816*u.Mpc**-3
agn_count_816 = agn_count_obs_816/f

#embed()

print("In NB527 (z=3.3) we predict: " + str(round(agn_count_obs_527.value,4)) + " unobstructed AGN and " + str(round(agn_count_527.value,4)) + " AGN total")
print("In NB718 (z=4.9) we predict: " + str(round(agn_count_obs_718.value,4)) + " unobstructed AGN and " + str(round(agn_count_718.value,4)) + " AGN total")
print("In NB816 (z=5.7) we predict: " + str(round(agn_count_obs_816.value,4)) + " unobstructed AGN and " + str(round(agn_count_816.value,4)) + " AGN total")




M_faint_vals = np.arange(-30,-20,0.01)
tot_count_obs_527 = np.zeros_like(M_faint_vals)
tot_count_obs_718 = np.zeros_like(M_faint_vals)
tot_count_obs_816 = np.zeros_like(M_faint_vals)

for i, M_faint in enumerate(M_faint_vals):
    lum_func_527 = quad(phi, -100, M_faint, args=(z_527, c_vals))[0]
    lum_func_718 = quad(phi, -100, M_faint, args=(z_718, c_vals))[0]
    lum_func_816 = quad(phi, -100, M_faint, args=(z_816, c_vals))[0]

    tot_count_obs_527[i] = vol_527.value * lum_func_527
    tot_count_obs_718[i] = vol_718.value * lum_func_718
    tot_count_obs_816[i] = vol_816.value * lum_func_816

tot_count_vals_527 = tot_count_obs_527/f
tot_count_vals_718 = tot_count_obs_718/f
tot_count_vals_816 = tot_count_obs_816/f


plt.plot(M_faint_vals,tot_count_obs_527,'r',label='NB527: Unobstructed AGN')
plt.plot(M_faint_vals,tot_count_vals_527,'r--',label='NB527: Total AGN')
plt.plot(M_faint_vals,tot_count_obs_718,'g',label='NB718: Unobstructed AGN')
plt.plot(M_faint_vals,tot_count_vals_718,'g--',label='NB718: Total AGN')
plt.plot(M_faint_vals,tot_count_obs_816,'b',label='NB816: Unobstructed AGN')
plt.plot(M_faint_vals,tot_count_vals_816,'b--',label='NB816: Total AGN')
plt.yscale("log")
plt.xlim(-20,-30)
plt.ylabel('AGN Count')
plt.xlabel('M_1450')
plt.grid()
plt.legend()
plt.show()



embed()