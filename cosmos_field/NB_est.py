import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
from astropy import units as u
from astropy import constants as c
from glob import glob
import csv
from IPython import embed


import numpy.random as rand
from astropy.table import Table
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit





NB527_CW = 5249.1 #Angstroms, Using value of highest transmittance from HSC Filter files
NB527_BW = 75 #Angstroms, Using difference in value of 0.5 transmittance from HSC Filter Files

NB718_CW = 7164.4 #Angstroms
NB718_BW = 110 #Angstroms
z_718 = (NB718_CW/1215.7) - 1

NB816_CW = 7170.5 #Angstroms
NB816_BW = 110 #Angstroms


L_718 = (1+z_718)*(c.c*NB718_BW / (Planck18.H(z_718)*7164.4)).to(u.Mpc)



def trans_sig_both(dist, data, noise_levels, boost_func, skewer_path, lambda_obs):
    N_pix_data = len(data)
    seed = 8986
    # weighting = norm(0,2)
    # weights = weighting.pdf(dist)

    data_sig = data.sum() / N_pix_data

    # Skewer Realizations

    params = Table.read(skewer_path, hdu=1)
    skewers = Table.read(skewer_path, hdu=2)

    Lbox_cMpc = params['Lbox'] / params['lit_h']
    drpix = Lbox_cMpc / params['Ng']
    rvec_cMpc = np.arange(params['Ng']) * drpix

    qso_idx = int(len(rvec_cMpc) / 2)
    rvec_cMpc -= rvec_cMpc[qso_idx]

    sim_boost = boost_func(rvec_cMpc)

    tau_skewers_wqso = skewers['TAU'] / sim_boost
    tau_skewers_nqso = skewers['TAU']

    trans_skewers_wqso = np.exp(-tau_skewers_wqso)
    trans_skewers_nqso = np.exp(-tau_skewers_nqso)

    # Using resolution convolution
    sig_coeff = 2*np.sqrt(2 * np.log(2))
    dlam = 0.65 # From keck deimos documentation for 600ZD
    sampling = (1.0 / 0.75) * 3.5 # Same ^^^
    fwhm = (dlam * 3e5 / lambda_obs) * sampling
    dvpix_hires = params['Ng']/params['VSIDE']
    pix_per_sigma = fwhm*dvpix_hires/sig_coeff

    convolved_trans_wqso  = gaussian_filter1d(trans_skewers_wqso,pix_per_sigma,axis=1,mode='wrap')
    convolved_trans_nqso  = gaussian_filter1d(trans_skewers_nqso,pix_per_sigma,axis=1,mode='wrap')


    summary_range = (rvec_cMpc >= dist[0]) & (rvec_cMpc <= dist[-1])
    trans_stat_wqso = convolved_trans_wqso[:, summary_range].sum(axis=1) / summary_range.sum()
    trans_stat_nqso = convolved_trans_nqso[:, summary_range].sum(axis=1) / summary_range.sum()

    rand.seed(seed)
    sig_vals_wqso = rand.normal(loc=0.0, scale=noise_levels, size=(1000, len(noise_levels))).sum(axis=1) / len(
        noise_levels)
    sig_vals_wqso += trans_stat_wqso
    sig_vals_wqso.sort()

    rand.seed(seed)
    sig_vals_nqso = rand.normal(loc=0.0, scale=noise_levels, size=(1000, len(noise_levels))).sum(axis=1) / len(
        noise_levels)
    sig_vals_nqso += trans_stat_nqso
    sig_vals_nqso.sort()

    y = np.arange(len(sig_vals_wqso)) / len(sig_vals_wqso)

    (mu_n, sigma_n) = norm.fit(sig_vals_nqso)
    (mu_w, sigma_w) = norm.fit(sig_vals_wqso)

    print(mu_w, mu_n)
    print(sigma_w, sigma_n)

    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, label='1')

    plt.vlines(data_sig, 0, 1, color='r', label='Data Significance')
    plt.vlines(0.45786, 0, 1, color='r', linestyles=':', label='Mean IGM Transmission: 0.458')


    plt.plot(sig_vals_wqso, y, color='g', label='Simulated Transmission Enhancement WITH Nearby Quasar')

    #w2lb_line = plt.Line2D(
    #    [sig_vals_wqso[int(len(sig_vals_wqso) * 0.025)], sig_vals_wqso[int(len(sig_vals_wqso) * 0.025)]],
    #    (0.025 - 0.03, 0.025 + 0.03), color='g', linestyle='--',
    #    lw=3, label='2 Sigma Lower Bound WITH Nearby Quasar')
    #w1lb_line = plt.Line2D(
    #    [sig_vals_wqso[int(len(sig_vals_wqso) * 0.16)], sig_vals_wqso[int(len(sig_vals_wqso) * 0.16)]],
    #    (0.16 - 0.03, 0.16 + 0.03), color='g', lw=3, label='1 Sigma Bounds WITH Nearby Quasar')

    w2lb_line = plt.Line2D(
        [sig_vals_wqso[int(len(sig_vals_wqso) * 0.025)], sig_vals_wqso[int(len(sig_vals_wqso) * 0.025)]],
        (0.025 - 0.03, 0.025 + 0.03), color='g', linestyle='--', lw=3)
    w1lb_line = plt.Line2D(
        [sig_vals_wqso[int(len(sig_vals_wqso) * 0.16)], sig_vals_wqso[int(len(sig_vals_wqso) * 0.16)]],
        (0.16 - 0.03, 0.16 + 0.03), color='g', lw=3)

    w1ub_line = plt.Line2D(
        [sig_vals_wqso[int(len(sig_vals_wqso) * 0.84)], sig_vals_wqso[int(len(sig_vals_wqso) * 0.84)]],
        (0.84 - 0.03, 0.84 + 0.03), color='g', lw=3)
    plt.gca().add_line(w2lb_line)
    plt.gca().add_line(w1lb_line)
    plt.gca().add_line(w1ub_line)




'''
# Currently Hard coded for Obj 8986

z = 4.169


comov_range =  #Create a comoving distance array


gamma_uvb = 1.03214425e-12 # Gamma_uvb from Nyx simulations;
gamma_qso_1cmpc = 2.62575329e-9 # Gamma_qso with input of 1 Mpc;

transverse_dist = 3.8996 # pMpc

wqso = gamma_qso_1cmpc/gamma_uvb

boost = (1.0 + wqso/(comov_range**2 + ((1+z_qso)*transverse_dist)**2)) # Uses Comoving distance

trans_mask = (vel_range>=-(qso_sig+50)) & (vel_range<=(qso_sig+50))
#trans_mask = (vel_range>=-(200)) & (vel_range<=(200))

nyx_skewer_path = "/home/sbechtel/Documents/software/enigma/enigma/tpe/Nyx_test/rand_skewers_z381_ovt_tau.fits"

boost_func = interp1d(comov_range,boost)

trans_sig_both(comov_range[trans_mask],trans_flux[trans_mask],trans_sig[trans_mask],
                                                  boost_func,nyx_skewer_path,qso_wave)
'''

files_list = glob("/home/sbechtel/Documents/software/enigma/enigma/tpe/Nyx_test/fred_z5_skewers/*.dat")

try:

    tau_vals = np.load("/home/sbechtel/Documents/software/enigma/enigma/tpe/Nyx_test/fred_z5_skewers/compiled.npz")['tau_vals']
    vel_vals = np.load("/home/sbechtel/Documents/software/enigma/enigma/tpe/Nyx_test/fred_z5_skewers/compiled.npz")['vel_vals']

except:

    tau_vals = np.zeros((1200,4096))

    vel_vals = []

    for i,file in enumerate(files_list):

        if i == len(files_list)-1:
            with open(file) as f:
                reader = csv.reader(f, delimiter=" ")
                for j,line in enumerate(reader):
                    vel_vals.append(float(line[0]))
                    tau_vals[i,j] = (float(line[6]))

        else:
            with open(file) as f:
                reader = csv.reader(f, delimiter=" ")
                for j,line in enumerate(reader):
                    tau_vals[i,j] = (float(line[6]))

    vel_vals = np.array(vel_vals)

    np.savez("/home/sbechtel/Documents/software/enigma/enigma/tpe/Nyx_test/fred_z5_skewers/compiled.npz", tau_vals=tau_vals, vel_vals=vel_vals)




embed()
