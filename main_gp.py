import os
#from cuda import cudart
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.98"

from jax import config
config.update("jax_enable_x64", True)

from scipy.stats import multivariate_normal as smn
import numpy as np
import jax.numpy as jnp
#ord_list = [[59,61]]
ord_list = [[60,61]]
######
# ord_list = [[56,66]]
######
mols = []
db = []
mols_l = ["H2O", "CH4", "CO"]
db_l = ["ExoMol", "HITEMP", "ExoMol"]
#####
#mols_l = ["CH4",]
#db_l = ["HITEMP"]
#####
for i in range(0, len(ord_list)): # order descending order just in case
    mols.append(mols_l)
    db.append(db_l)

num_ord_list_p = 1
mols_p = [mols_l]
db_p = [db_l]


import moldata, setting
db_dir = moldata.set_db_dir(mols, db)
db_dir_p = moldata.set_db_dir(mols_p, db_p)
path_obs, path_data, path_repo = setting.set_path()
#setting.git_install(path_repo, "master")



import obs
# [A], f_nu
ld_obs, f_obs, f_obserr, ord, f_ref = obs.spec(path_obs, ord_list, ord_norm=60)
H_mag_obs = 13.04 # https://www.aanda.org/articles/aa/abs/2003/40/aa3808/aa3808.html
H_mag_obserr = 0.10

nusd = []
for k in range(len(ord_list)):
    nusd_k = jnp.array(1.0e8/ld_obs[k]) #wavenumber
    nusd.append(nusd_k)



from exojax.atm.atmprof import pressure_layer_logspace
NP=200
Parr, dParr, k=pressure_layer_logspace(nlayer=NP, log_pressure_top=-4)
ONEARR=np.ones_like(Parr)

from exojax.utils.grids import wavenumber_grid
import math
R = 300000. # 10 x instrumental spectral resolution
nus = []
wav = []
res = []
for k in range(len(ord_list)):
    nu_min = 1.0e8/(np.max(ld_obs[k])+5.0)
    nu_max = 1.0e8/(np.min(ld_obs[k])-5.0)
    Nx = math.ceil(R * np.log(nu_max/nu_min)) + 1 # ueki
    Nx = math.ceil(Nx/2.) * 2 # make even
    nus_k,wav_k,res_k=wavenumber_grid(np.min(ld_obs[k])-5.0,np.max(ld_obs[k])+5.0,Nx,unit="AA",xsmode="premodit")
    print(len(nus_k), res_k)
    nus.append(nus_k)
    wav.append(wav_k[::-1])
    res.append(res_k)



# photometry
d = np.genfromtxt(os.path.join(path_obs, "Keck_NIRC2.Ks.dat"))
wl_ref = d[:,0]
tr_ref = d[:,1]

wl_min = np.min(wl_ref)
wl_max = np.max(wl_ref)
dlmd = (wl_max - wl_min) / len(wl_ref)
Rinst_p = 0.5 * (wl_min + wl_max) / dlmd
#####
# wl_min = np.min(1.0e8/np.concatenate(nusd))
# wl_max = np.max(1.0e8/np.concatenate(nusd))
# Rinst_p = 3257
#####
print(Rinst_p)
R = Rinst_p * 2.**5 # 10 x instrumental spectral resolution
nus_p = []
wav_p = []
res_p = []
nusd_p = []
wavd_p = []
for k in range(num_ord_list_p):
    nu_min = 1.0e8/(wl_max + 5.0)
    nu_max = 1.0e8/(wl_min - 5.0)
    Nx = math.ceil(R * np.log(nu_max/nu_min)) + 1 # ueki
    Nx = math.ceil(Nx/2.) * 2 # make even
    nus_k,wav_k,res_k = wavenumber_grid(wl_min-5.,wl_max+5.,Nx,unit="AA",xsmode="premodit")
    print(len(nus_k), res_k)
    nus_p.append(nus_k)
    wav_p.append(wav_k[::-1])
    res_p.append(res_k)

    mask_p = (1.0e8/nus_k >= wl_min) * (1.0e8/nus_k <= wl_max)
    nusd_p.append(nus_k[mask_p])
    wavd_p.append(1.0e8/nus_k[mask_p])

from scipy import interpolate
f = interpolate.interp1d(wl_ref, tr_ref)
tr = f(wavd_p)
import matplotlib.pyplot as plt
# plt.plot(wl_ref, tr_ref)
# plt.plot(np.concatenate(wavd_p), np.concatenate(tr))
# plt.show()



mols, db, db_dir, mdb = moldata.set_mdb(path_data, mols, db, db_dir, nus, crit=1.e-30, Ttyp=1000.)
opa = moldata.set_opa_premodit(mols, mdb, nus, T_min=300., T_max=3000.)

from exojax.spec import contdb
cdbH2H2=[]
cdbH2He=[]
for k in range(len(ord_list)):
    cdbH2H2.append(contdb.CdbCIA(os.path.join(path_data,'H2-H2_2011.cia'),nus[k]))
    cdbH2He.append(contdb.CdbCIA(os.path.join(path_data,'H2-He_2011.cia'),nus[k]))

mols_unique, mols_num, molmass, molmassH2, molmassHe = moldata.set_molmass(mols)


mols_p, db_p, db_dir_p, mdb_p = moldata.set_mdb(path_data, mols_p, db_p, db_dir_p, nus_p, crit=1.e-30, Ttyp=1000.)
opa_p = moldata.set_opa_premodit(mols_p, mdb_p, nus_p, T_min=300., T_max=3000.)

cdbH2H2_p=[]
cdbH2He_p=[]
for k in range(num_ord_list_p):
    cdbH2H2_p.append(contdb.CdbCIA(os.path.join(path_data,'H2-H2_2011.cia'),nus_p[k]))
    cdbH2He_p.append(contdb.CdbCIA(os.path.join(path_data,'H2-He_2011.cia'),nus_p[k]))

mols_num_p = moldata.mols_num_p(mols_p, mols_unique)



from astropy import constants as const
Mjup = const.M_jup.value
Rjup = const.R_jup.value
G_const = const.G.value
pc = const.pc.value

from exojax.spec.layeropacity import layer_optical_depth, layer_optical_depth_CIA
from exojax.spec.atmrt import ArtEmisPure
from exojax.spec import response

from exojax.utils.instfunc import R2STD
Rinst=30000. #instrumental spectral resolution
beta_inst=R2STD(Rinst)
beta_inst_p=R2STD(Rinst_p)

# from modit_hitran import hitran
from jax import vmap

# response settings
from exojax.utils.grids import velocity_grid
from exojax.spec.spin_rotation import convolve_rigid_rotation
vsini_max = 100.0
vr_array = []
for k in range(len(ord_list)):
    vr_array.append(velocity_grid(res[k], vsini_max))

def frun(T0, alpha, logg, Mp, logvmr, u1, u2, RV, vsini, onl=None):
    Tarr = T0*(Parr)**alpha
    Tarr = jnp.clip(Tarr, 300, None)

    g = 10.**logg # cgs
    g_mks = g * 1e-2
    Rp = jnp.sqrt(G_const * Mp * Mjup / g_mks) / Rjup

    vmr = jnp.power(10., jnp.array(logvmr))
    vmrH2 = (1. - jnp.sum(vmr)) * 6./7.
    vmrHe = (1. - jnp.sum(vmr)) * 1./7.
    mmw = jnp.sum(vmr*jnp.array(molmass)) + vmrH2*molmassH2 + vmrHe*molmassHe
    mmr = jnp.multiply(vmr, jnp.array(molmass)) / mmw

    mu = []
    for k in range(len(ord_list)):
        art = ArtEmisPure(pressure_top=1.e-4,
                          pressure_btm=1.e2,
                          nlayer=200,
                          nu_grid=nus[k],
                          rtsolver="ibased",
                          nstream=8)

        dtaum = []
        for i in range(len(mols[k])):
            if(onl is None or mols[k][i] == onl):
                xsm = opa[k][i].xsmatrix(Tarr, Parr)
                xsm = jnp.abs(xsm)
                dtaum.append(layer_optical_depth(dParr,xsm,mmr[mols_num[k][i]]*ONEARR,molmass[mols_num[k][i]],g))

        dtau = sum(dtaum)

        #CIA
        if(len(cdbH2H2[k].nucia) > 0):
            dtaucH2H2 = layer_optical_depth_CIA(nus[k],Tarr,Parr,dParr,vmrH2,vmrH2,mmw,g,cdbH2H2[k].nucia,cdbH2H2[k].tcia,cdbH2H2[k].logac)
            dtau = dtau + dtaucH2H2
        if(len(cdbH2He[k].nucia) > 0):
            dtaucH2He = layer_optical_depth_CIA(nus[k],Tarr,Parr,dParr,vmrH2,vmrHe,mmw,g,cdbH2He[k].nucia,cdbH2He[k].tcia,cdbH2He[k].logac)
            dtau = dtau + dtaucH2He

        F0 = art.run(dtau,Tarr)

        Frot = convolve_rigid_rotation(F0, vr_array[k], vsini, u1, u2)
        mu_k = response.ipgauss_sampling(nusd[k],nus[k],Frot,beta_inst,RV,vr_array[k])

        #distance correction from Kirkpatrick et al. 2012
        mu_k = mu_k * ((Rp * Rjup) /(17.72 * pc))**2
        # account for normalization
        mu_k = mu_k / f_ref

        mu.append(mu_k)

    return jnp.array(mu)



#f0 = 1.15e-9 # [W/m^2/um] # kurohon, checked the referenced book in google book
f0 = 4.29e-10 # https://irsa.ipac.caltech.edu/data/SPITZER/docs/dataanalysistools/tools/pet/magtojy/index.html 2MASS Ks
def calc_photo(mu):
    mu = jnp.concatenate(mu)
    mu = mu * f_ref # [erg/s/cm^2/cm^{-1}]
    # [erg/s/cm^2/cm^{-1}] => [erg/s/cm^2/cm]
    mu = mu / (jnp.concatenate(wavd_p)*1.0e-8)**2.0e0
    # [erg/s/cm^2/cm] => [W/m^2/um]
    mu = mu * 1.0e-7 * 1.0e4 * 1.0e-4

    fdl = jnp.trapezoid(mu*jnp.concatenate(tr), jnp.concatenate(wavd_p))
    dl = jnp.trapezoid(jnp.concatenate(tr), jnp.concatenate(wavd_p))
    f = fdl / dl

    H_mag = -2.5 * jnp.log10(f / f0)

    return H_mag


vsini_max = 100.0
vr_array_p = []
for k in range(num_ord_list_p):
    vr_array_p.append(velocity_grid(res_p[k], vsini_max))

def frun_p(T0, alpha, logg, Mp, logvmr, u1, u2, RV, vsini):
    Tarr = T0*(Parr)**alpha
    Tarr = jnp.clip(Tarr, 300, None)

    g = 10.**logg # cgs
    g_mks = g * 1e-2
    Rp = jnp.sqrt(G_const * Mp * Mjup / g_mks) / Rjup

    vmr = jnp.power(10., jnp.array(logvmr))
    vmrH2 = (1. - jnp.sum(vmr)) * 6./7.
    vmrHe = (1. - jnp.sum(vmr)) * 1./7.
    mmw = jnp.sum(vmr*jnp.array(molmass)) + vmrH2*molmassH2 + vmrHe*molmassHe
    mmr = jnp.multiply(vmr, jnp.array(molmass)) / mmw

    mu = []
    for k in range(num_ord_list_p):
        art = ArtEmisPure(pressure_top=1.e-4,
                          pressure_btm=1.e2,
                          nlayer=200,
                          nu_grid=nus_p[k],
                          rtsolver="ibased",
                          nstream=8)

        dtaum = []
        for i in range(len(mols_p[k])):
            xsm = opa_p[k][i].xsmatrix(Tarr, Parr)
            xsm = jnp.abs(xsm)
            dtaum.append(layer_optical_depth(dParr,xsm,mmr[mols_num_p[k][i]]*ONEARR,molmass[mols_num_p[k][i]],g))

        dtau = sum(dtaum)

        #CIA
        if(len(cdbH2H2[k].nucia) > 0):
            dtaucH2H2 = layer_optical_depth_CIA(nus_p[k],Tarr,Parr,dParr,vmrH2,vmrH2,mmw,g,cdbH2H2_p[k].nucia,cdbH2H2_p[k].tcia,cdbH2H2_p[k].logac)
            dtau = dtau + dtaucH2H2
        if(len(cdbH2He[k].nucia) > 0):
            dtaucH2He = layer_optical_depth_CIA(nus_p[k],Tarr,Parr,dParr,vmrH2,vmrHe,mmw,g,cdbH2He_p[k].nucia,cdbH2He_p[k].tcia,cdbH2He_p[k].logac)
            dtau = dtau + dtaucH2He

        F0 = art.run(dtau,Tarr)

        Frot = convolve_rigid_rotation(F0, vr_array_p[k], vsini, u1, u2)
        mu_k = response.ipgauss_sampling(nusd_p[k],nus_p[k],Frot,beta_inst_p,RV,vr_array_p[k])

        #distance correction from Kirkpatrick et al. 2012
        mu_k = mu_k * ((Rp * Rjup) /(17.72 * pc))**2
        # account for normalization
        mu_k = mu_k / f_ref

        mu.append(mu_k)

    H_mag = calc_photo(mu)
    return H_mag



logvmr_sample = [-3.6, -7.90, -3.34]
logvmr_sample = logvmr_sample[0:len(mols_unique)]
mu = frun(T0=1500.00, alpha=0.09, logg=4.92, Mp=72.7, logvmr=logvmr_sample, \
          u1=0.0, u2=0.0, RV=5.99, vsini=40.0)
print(mu)
H_mag = frun_p(T0=1500.0, alpha=0.09, logg=4.92, Mp=72.7, logvmr=logvmr_sample, \
               u1=0.0, u2=0.0, RV=5.99, vsini=40.0)
print(H_mag)



def objective(param, f_model, err_all):
    f = (param * jnp.concatenate(f_model) - jnp.concatenate(f_obs)) / err_all
    cost = jnp.dot(f,f)
    return cost

from jaxopt import OptaxSolver
import optax
adam = OptaxSolver(opt=optax.adam(2.e-2), fun=objective)
param, state = adam.run(init_params=1., f_model=mu, err_all=jnp.concatenate(f_obserr))
print(param)
# plt.plot(1.0e8/jnp.concatenate(nusd), jnp.concatenate(f_obs),"+", c="k")
# plt.plot(1.0e8/jnp.concatenate(nusd), jnp.concatenate(mu), c="C0")
# plt.plot(1.0e8/jnp.concatenate(nusd), jnp.concatenate(mu) * param, c="C1")
# plt.show()
# plt.close()



from jax import random
import numpyro.distributions as dist
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
from numpyro.diagnostics import hpdi
from exojax.utils.gpkernel import gpkernel_RBF

def model_c(nusd,y1,y1err,y2,y2err,onl=None):
    logg = numpyro.sample('logg', dist.TruncatedNormal(5.5,0.2,low=5.)) # Wang et al. (2022)
    #logg = numpyro.sample('logg', dist.Uniform(4.0,6.0))
    Mp = numpyro.sample('Mp', dist.TruncatedNormal(72.7,0.8,low=1.)) # Brandt et al. (2021)
    RV = numpyro.sample('RV', dist.Uniform(-10,10))
    T0 = numpyro.sample('T0', dist.Uniform(500.0,2500.0))
    alpha = numpyro.sample('alpha', dist.Uniform(0.0,0.2))
    vsini = numpyro.sample('vsini', dist.Uniform(0.0,40.0))
    logvmr = []
    for i in range(len(mols_unique)):
        logvmr.append(numpyro.sample('log'+mols_unique[i], dist.Uniform(-10.0,0.0)))
    u1 = 0.0
    u2 = 0.0

    #sigma = numpyro.sample('sigma',dist.Exponential(10.0))
    #sig = jnp.ones_like(jnp.concatenate(nusd)) * sigma
    #err_all = jnp.sqrt(y1err**2. + sig**2.)

    # Sample parameters for the Gaussian Process kernel
    logtau = numpyro.sample('logtau', dist.Uniform(-1.5, 0.5))
    tau = 10**logtau
    loga = numpyro.sample('loga', dist.Uniform(-4.5, -2.0))
    a = 10**loga
    # Define the covariance matrix for the Gaussian Process
    cov = gpkernel_RBF(jnp.concatenate(nusd), tau, a, y1err)

    mu = frun(T0, alpha, logg, Mp, logvmr, u1, u2, RV, vsini)
    # param, state = adam.run(init_params=1., f_model=mu, err_all=err_all)
    param = jnp.dot(jnp.concatenate(mu)/y1err, jnp.concatenate(f_obs)/y1err) \
        / jnp.dot(jnp.concatenate(mu)/y1err, jnp.concatenate(mu)/y1err)
    a = numpyro.deterministic("a", param)
    if onl is not None:
        mu = frun(T0, alpha, logg, Mp, logvmr, u1, u2, RV, vsini, onl=onl)
    mu_all = jnp.concatenate(mu) * param
    H_mag = frun_p(T0, alpha, logg, Mp, logvmr, u1, u2, RV, vsini)

    if False:
        # GP Likelihood
        numpyro.sample("y1", dist.MultivariateNormal(loc=mu_all, covariance_matrix=cov), obs=y1)
        numpyro.sample("y2", dist.Normal(H_mag, y2err), obs=y2)
    else:
        # Weighting Factors
        spec_weight = 1.0
        phot_weight = 1e4  # Increase to balance photometric bias

        log_prob_y1 = jnp.sum(dist.MultivariateNormal(loc=mu_all, covariance_matrix=cov).log_prob(y1))
        log_prob_y2 = dist.Normal(H_mag, y2err).log_prob(y2)

        # Apply the weight in log-space correctly
        numpyro.factor("spec_log_likelihood", log_prob_y1 + jnp.log(spec_weight))
        numpyro.factor("phot_log_likelihood", log_prob_y2 + jnp.log(phot_weight))

mcmc_flag = True
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 25, 100
######
# num_warmup, num_samples = 100, 200
######
if mcmc_flag:
    kernel = NUTS(model_c,forward_mode_differentiation=True)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    import time
    t1 = time.perf_counter()
    mcmc.run(rng_key_, nusd=nusd, y1=jnp.concatenate(f_obs), y1err=jnp.concatenate(f_obserr), \
             y2=H_mag_obs, y2err=H_mag_obserr)
    t2 = time.perf_counter()
    print(t2 - t1)
    mcmc.print_summary()

# Define the Gaussian Process kernel for prediction
def gpkernel_GK_x(t, t2, tau, a):
    Dt = t - jnp.array([t2]).T
    return a * jnp.exp(-0.5 * (Dt / tau) ** 2)

# Predict using the GP and the sampled parameters

def prediction(t_pred, t, y, samples, idx, err):
    y = jnp.array(y).flatten()
    T0, alpha, logg, Mp, u1, u2, RV, vsini = samples['T0'][idx], samples['alpha'][idx], samples['logg'][idx], samples['Mp'][idx], 0.0, 0.0, samples['RV'][idx], samples['vsini'][idx]
    tau, a = 10 ** samples['logtau'][idx], 10 ** samples['loga'][idx]
    

    logvmr = []
    for i in range(len(mols_unique)):
        logvmr.append(samples['log'+mols_unique[i]][idx])

    mu = frun(T0, alpha, logg, Mp, logvmr, u1, u2, RV, vsini)
    mu = mu.flatten()
    param = samples['a'][idx]
    mu = mu * param
    # GP Covariance Matrices
    Ksigma = gpkernel_RBF(t, tau, a, err)
    Kpred = gpkernel_RBF(t_pred, tau, a, err)
    Kx = gpkernel_GK_x(t, t_pred, tau, a)

    #print("mu shape: ", mu.shape)
    #print("Kx.T shape: ", Kx.T.shape)
    #print("Ksigma shape: ", Ksigma.shape)
    #print("y shape: ", y.shape)
    #print("(y - mu) shape: ", (y - mu).shape)

    y_pred = mu + Kx.T @ jnp.linalg.inv(Ksigma) @ (y - mu)
    cov_pred = Kpred - Kx.T @ jnp.linalg.inv(Ksigma) @ Kx
    #return y_pred, cov_pred, mu/gpkernel_GK_x
    return y_pred, cov_pred, mu

if True:
    import pickle
    if mcmc_flag:
        import arviz as az
        az.plot_trace(mcmc, backend_kwargs={"constrained_layout":True})
        plt.savefig("./output_gp/trace.pdf", bbox_inches='tight')

        with open("./output_gp_bn/mcmc.pickle", mode='wb') as f:
            pickle.dump(mcmc, f)
        with open("./output_gp_bn/samples.pickle", mode='wb') as f:
            pickle.dump(mcmc.get_samples(), f)



    with open("./output_gp_bn/samples.pickle", mode='rb') as f:
        samples = pickle.load(f)

    marr = []
    for idx in jnp.array(np.random.randint(0, num_samples, 100)):
        y_pred, cov_pred, mu = prediction(jnp.concatenate(nusd), jnp.concatenate(nusd), jnp.concatenate(f_obs), samples, idx, jnp.concatenate(f_obserr))
        mk = smn(mean=y_pred, cov=cov_pred, allow_singular=True).rvs(1).T
        marr.append(mk)


    from numpyro.diagnostics import hpdi

    marr = jnp.array(marr)
    print("marr shape: ", jnp.shape(marr))
    mean_marr = np.mean(marr, axis=0)
    hpdi_marr = hpdi(marr, 0.95)

    #pred = Predictive(model_c,samples,return_sites=["y1","y2"])
    #predictions = pred(rng_key_,nusd=nusd,y1=None,y1err=jnp.concatenate(f_obserr), \
    #                   y2=None, y2err=H_mag_obserr)

    median_mu1 = jnp.array(mean_marr)
    hpdi_mu1 = jnp.array(hpdi_marr)
    print(np.shape(median_mu1), np.shape(hpdi_mu1))

    with open("./output_gp/pred.pickle", mode='wb') as f:
        pickle.dump(marr, f)

    np.savez("./output_gp/hpdi_mu1.npz", hpdi_mu1)

    import matplotlib.pyplot as plt
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
    plt.switch_backend('agg')
    i_min = 0
    i_max = 0
    for k in range(len(ord_list)):
        i_max = i_max + len(ld_obs[k])
        print(i_min, i_max, np.shape(ld_obs))
        fig, ax = plt.subplots(figsize=(28,6.0))
        ax.plot(ld_obs[k],f_obs[k],"+",color="black",label="data")
        ax.plot(ld_obs[k],median_mu1,color="C0",label="median")
        ax.fill_between(ld_obs[k], hpdi_mu1[0], hpdi_mu1[1], alpha=0.3, interpolate=True,color="C0",label="95% area")

        plt.xlabel("Wavelength [$\AA$]", fontsize=15)
        plt.ylabel("Normalized Flux", fontsize=15)
        ax.set_xlim(np.min(ld_obs[k]), np.max(ld_obs[k]))

        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        plt.legend(fontsize=16)
        plt.tick_params(labelsize=16)
        # plt.show()

        num = str(ord_list[k][0]) + "-" + str(ord_list[k][1])
        plt.set_ylim(0.6, 1.3)
        plt.savefig("./output_gp/fit"+num+".pdf", bbox_inches='tight')
        plt.close()

        i_min = i_max

    wavd = jnp.array(ld_obs).flatten()
    flux = jnp.array(f_obs).flatten()
    err = jnp.array(f_obserr).flatten()

    residuals_gp = flux - median_mu1

    # Calculate percentiles for residuals with Gaussian Processes (GP)
    lower_bound_gp, upper_bound_gp = np.percentile(residuals_gp, [2.5, 97.5])

    from matplotlib import gridspec

    fig = plt.figure(figsize=(20, 6.0))
    spec = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[4, 1.5], hspace=0.07)

    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])

    ax0.plot(wavd, flux, ".", color='black', label="data", ms=5)
    ax0.plot(wavd, median_mu1, color='C0', label="fit w/ GP")
    ax0.fill_between(wavd, (hpdi_marr[0]), (hpdi_marr[1]), alpha=0.5, interpolate=True, color='C0', label='95% area w/ GP')
    # ax0.set_xlim(22300, 22700)

    # Plotting residuals with GP
    ax1.errorbar(wavd, residuals_gp, yerr=err, fmt='o', markersize=3, color='C0', label="residual w/ GP", alpha=0.5)
    ax1.set_ylim(lower_bound_gp - 0.01, upper_bound_gp + 0.01)  # Add a small margin
    ax1.legend(fontsize=12)
    ax1.tick_params(labelsize=12)

    ax0.axes.xaxis.set_visible(False)
    ax0.legend(fontsize=12)
    ax0.tick_params(labelsize=12)
    ax1.set_xlabel('wavelength ($\AA$)', fontsize=16)

    lower = 22900
    upper = 23250

    # ax0.set_xlim(lower, upper)
    # ax1.set_xlim(lower, upper)
    # ax2.set_xlim(lower, upper)

    ax0.set_ylim(0.6, 1.3)

    plt.savefig("./output_gp/flux_residual.png", bbox_inches='tight')

    import corner
    figure = corner.corner(samples, show_titles=True, quantiles=[0.16, 0.5, 0.84], color='C0', label_kwargs={"fontsize": 20}, smooth=1.0) # smooth parameter needed
    figure.savefig("./output_gp/corner.pdf", bbox_inches='tight')

