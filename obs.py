import numpy as np
import os
def spec(path_obs, ord_list, ord_norm):
    d = np.genfromtxt(os.path.join(path_obs, "hr7672b_20240108.dat"))
    d = d[np.argsort(d[:, 0])]
    # [um], f_lmd
    ld_obs, f_obs_lmd, f_obserr_lmd, ord = d[:,0], d[:,1], d[:, 2], d[:, 3]
    ord = np.array(ord, dtype='int')

    # [um] => [A]
    ld_obs = ld_obs * 1.0e4

    # f_obs_lmd => f_obs_nu
    f_obs_nu = f_obs_lmd * (ld_obs)**2.0e0
    f_obserr_nu = f_obserr_lmd * (ld_obs)**2.0e0


    # normalize by the median wavelength of ord_norm
    mask = (ord==int(ord_norm))
    ld0 = np.median(ld_obs[mask])
    f_itp = interpolate.interp1d(ld_obs, f_obs_nu, kind='linear')
    f_obs0 = f_itp(ld0)

    # normalize by the flux at ld0
    f_obs_nu = f_obs_nu / f_obs0
    f_obserr_nu = f_obserr_nu / f_obs0

    f_ref = norm_flux(path_obs, ld0)

    ld_obs_l = []
    f_obs_nu_l = []
    f_obserr_nu_l = []
    ord_l = []
    for k in range(len(ord_list)):
        ord0_max = np.where(ord==int(ord_list[k][0]-1))[0][-1]
        ord2_max = np.where(ord==ord_list[k][1])[0][-1]
        # order masking
        mask = (ld_obs > ld_obs[ord0_max]) * (ld_obs <= ld_obs[ord2_max])
        ld_obs_k = ld_obs[mask]
        f_obs_nu_k = f_obs_nu[mask]
        f_obserr_nu_k = f_obserr_nu[mask]
        ord_k = ord[mask]

        # wavelength descending order
        ld_obs_k = ld_obs_k[::-1]
        f_obs_nu_k = f_obs_nu_k[::-1]
        f_obserr_nu_k = f_obserr_nu_k[::-1]
        ord_k = ord_k[::-1]

        ld_obs_l.append(ld_obs_k)
        f_obs_nu_l.append(f_obs_nu_k)
        f_obserr_nu_l.append(f_obserr_nu_k)
        ord_l.append(ord_k)

    return ld_obs_l, f_obs_nu_l, f_obserr_nu_l, ord_l, f_ref



from scipy import interpolate
def norm_flux(path_obs, ld0):
    if False:
        d = np.genfromtxt(os.path.join(path_obs, "T7.txt"))
        # [um], [W/m^2/um]
        ld_ref, f_ref = d[:,0], d[:,1]
    
        # [um] => [A]
        ld_ref = ld_ref * 1.0e4
    
        # [W/m^2/um] => [erg/s/cm^2/cm]
        f_ref = f_ref * 1.0e7 * 1.0e-4 * 1.0e4
        # [erg/s/cm^2/cm] => [erg/s/cm^2/cm^{-1}] # units of exojax
        f_ref = f_ref * (ld_ref * 1.0e-8)**2.0e0
    
        mask = (f_ref > 0.)
        f_itp = interpolate.interp1d(ld_ref[mask], f_ref[mask], kind='linear')
        f0 = f_itp(ld0) # ref flux at ld0
    else:
        ld_ref = 2.159
        ld_ref = ld_ref * 1.0e4
        f_ref = 1.4e-15
        f_ref = f_ref * 1.0e7 * 1.0e-4 * 1.0e4
        f_ref = f_ref * (ld_ref * 1.0e-8)**2.0e0
        f0 = f_ref
    return f0
