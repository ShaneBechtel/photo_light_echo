import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
from astropy import units as u


# Get npz files

nb527 = np.load("NB527_mag_dist.npz")
nb718 = np.load("NB718_mag_dist.npz")
nb816 = np.load("NB816_mag_dist.npz")


ang_527 = 2.2 * u.arcmin
ang_718 = 4.8 * u.arcmin
ang_816 = 12.6 * u.arcmin


mean_sep_527 = (1 + 3.3) * Planck18.angular_diameter_distance(3.3) * ang_527.to(u.rad).value
mean_sep_718 = (1 + 4.9) * Planck18.angular_diameter_distance(4.9) * ang_718.to(u.rad).value
mean_sep_816 = (1 + 5.7) * Planck18.angular_diameter_distance(5.7) * ang_816.to(u.rad).value

# Make Figure

plt.figure(figsize=(15,10))

plt.title("(CURRENTLY DO NOT TRUST NB718 & NB816 SIMS)")

plt.plot(nb527['M1450'], nb527['DIST'], 'r', label='Distance to Gamma_QSO = Gamma_UVB in NB527 (z=3.3)')
plt.plot(nb718['M1450'], nb718['DIST'], 'g', label='Distance to Gamma_QSO = Gamma_UVB in NB718 (z=4.9)')
plt.plot(nb816['M1450'], nb816['DIST'], 'b', label='Distance to Gamma_QSO = Gamma_UVB in NB816 (z=5.7)')

plt.hlines(mean_sep_527.value,-22,-27, 'r', linestyles='--', label='Mean Sightline Separation in NB527 (z=3.3)')
plt.hlines(mean_sep_718.value,-22,-27, 'g', linestyles='--', label='Mean Sightline Separation in NB718 (z=4.9)')
plt.hlines(mean_sep_816.value,-22,-27, 'b', linestyles='--', label='Mean Sightline Separation in NB816 (z=5.7)')

plt.xlim(-22, -27)
plt.xlabel(r'M_{1450}')
plt.ylabel('Distance (pMpc)')
plt.legend()
plt.show()