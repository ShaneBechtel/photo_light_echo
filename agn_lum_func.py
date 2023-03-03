import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebval as cval
from astropy.cosmology import Planck18
from astropy import units as u
from astropy import constants as c
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

    return phi_star/denom #Log_10(phi/cMpc**-3 * mag**-1)


#Volume of survey
z_33 = (5249.1/1215.7) - 1 #Using value of highest transmittance from HSC Filter files
z_33_low = (5215.5/1215.7) - 1 #Using lower value of 0.5 transmittance from HSC Filter Files
z_33_high = (5291.2/1215.7) - 1 #Using lower value of 0.5 transmittance from HSC Filter Files

z_49 = (7164.4/1215.7) - 1
z_49_low = (7111.9/1215.7) - 1
z_49_high = (7221.4/1215.7) - 1

z_57 = (8157.9/1215.7) - 1
z_57_low = (8113.4/1215.7) - 1
z_57_high = (8224.7/1215.7) - 1


ang_radius = 1.67/2 #degrees, area of Cosmos field

ang_dist_33 = Planck18.angular_diameter_distance(z_33)
length_33 = Planck18.comoving_distance(z_33_high)-Planck18.comoving_distance(z_33_low)
vol_comov_33 = np.pi * (ang_radius * (np.pi/180) * ang_dist_33)**2 * length_33
vol_33 = vol_comov_33 * (1+z_33)**3

ang_dist_49 = Planck18.angular_diameter_distance(z_49)
length_49 = Planck18.comoving_distance(z_49_high)-Planck18.comoving_distance(z_49_low)
vol_comov_49 = np.pi * (ang_radius * (np.pi/180) * ang_dist_49)**2 * length_49
vol_49 = vol_comov_49 * (1+z_49)**3

ang_dist_57 = Planck18.angular_diameter_distance(z_57)
length_57 = Planck18.comoving_distance(z_57_high)-Planck18.comoving_distance(z_57_low)
vol_comov_57 = np.pi * (ang_radius * (np.pi/180) * ang_dist_57)**2 * length_57
vol_57 = vol_comov_57 * (1+z_57)**3


f=1-np.cos(np.pi/6) #Assume opening angle of 30 degs

#lower limit of -25 for mag
lum_func_33 = phi(-25,z_33,c_vals) #Evaluate luminosity function at -25 Mag
agn_count_obs_33 = vol_33*lum_func_33*u.Mpc**-3 #Obstructed AGN count
agn_count_33 = agn_count_obs_33/f #Unobstructed AGN count

lum_func_49 = phi(-25,z_49,c_vals)
agn_count_obs_49 = vol_49*lum_func_49*u.Mpc**-3
agn_count_49 = agn_count_obs_49/f

lum_func_57 = phi(-25,z_57,c_vals)
agn_count_obs_57 = vol_57*lum_func_57*u.Mpc**-3
agn_count_57 = agn_count_obs_57/f

#embed()

print("In NB527 (z=3.3) we predict: " + str(round(agn_count_obs_33.value,4)) + " unobstructed AGN and " + str(round(agn_count_33.value,4)) + " AGN total")
print("In NB718 (z=4.9) we predict: " + str(round(agn_count_obs_49.value,4)) + " unobstructed AGN and " + str(round(agn_count_49.value,4)) + " AGN total")
print("In NB816 (z=5.7) we predict: " + str(round(agn_count_obs_57.value,4)) + " unobstructed AGN and " + str(round(agn_count_57.value,4)) + " AGN total")