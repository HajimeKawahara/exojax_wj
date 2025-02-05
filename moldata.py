import numpy as np

from exojax.spec.multimol import database_path_hitran12
from exojax.spec.multimol import database_path_hitemp
from exojax.spec.multimol import database_path_exomol
def set_db_dir(mols, db):
    db_dir = []
    for k in range(len(mols)):
        db_dir_k = []
        for i in range(len(mols[k])):
            if(db[k][i] == "ExoMol"):
                # db_dir_k.append(database_path_exomol(mols[k][i]))
                if(mols[k][i] == "H2O"):
                    db_dir_k.append("H2O/1H2-16O/POKAZATEL")
                if(mols[k][i] == "CO"):
                    db_dir_k.append("CO/12C-16O/Li2015")
                elif(mols[k][i] == "NH3"):
                    db_dir_k.append("NH3/14N-1H3/CoYuTe")
            elif(db[k][i] == "HITRAN"):
                db_dir_k.append(database_path_hitran12(mols[k][i]))
            elif(db[k][i] == "HITEMP"):
                db_dir_k.append(database_path_hitemp(mols[k][i]))

        if None in db_dir_k:
            print("db_dir not specified")
            exit()
        db_dir.append(db_dir_k)

    return db_dir



from exojax.spec import api
import os
def set_mdb(path_data, mols, db, db_dir, nus, crit=0., Ttyp=1000.):
    mdb = []
    for k in range(len(mols)):
        mdb_k = []
        for i in range(len(mols[k])):
            if(db[k][i] == "ExoMol"):
                mdb_k.append(api.MdbExomol(os.path.join(path_data,db_dir[k][i]),nus[k],crit=crit,Ttyp=Ttyp,gpu_transfer=False))
            elif(db[k][i] == "HITRAN"):
                mdb_k.append(api.MdbHitran(os.path.join(path_data,db_dir[k][i]),nus[k],crit=crit,Ttyp=Ttyp,gpu_transfer=False, isotope=1))
                # print(mdb_k[i].exact_isotope_name(1))
            elif(db[k][i] == "HITEMP"):
                mdb_k.append(api.MdbHitemp(os.path.join(path_data,db_dir[k][i]),nus[k],crit=crit,Ttyp=Ttyp,gpu_transfer=False, isotope=1))
                print(mdb_k[i].exact_isotope_name(1))

        # remove the species with zero lines
        idx_del = []
        for i in range(len(mols[k])):
            print(mols[k][i], len(mdb_k[i].nu_lines)/1.e4, "mann")
            if(len(mdb_k[i].nu_lines) == 0):
                idx_del.append(i)

        for i in range(len(idx_del)):
            del mols[k][idx_del[i]]
            del db[k][idx_del[i]]
            del db_dir[k][idx_del[i]]
            del mdb_k[idx_del[i]]

        mdb.append(mdb_k)

    return mols, db, db_dir, mdb



from exojax.spec import molinfo
def set_molmass(mols):
    mols_unique = []
    mols_num = []
    for k in range(len(mols)):
        mols_num_k = []
        for i in range(len(mols[k])):
            if mols[k][i] in mols_unique:
                mols_num_k.append(mols_unique.index(mols[k][i]))
            else:
                mols_unique.append(mols[k][i])
                mols_num_k.append(mols_unique.index(mols[k][i]))
        mols_num.append(mols_num_k)

    molmass = []
    for i in range(len(mols_unique)):
        print(mols_unique[i], molinfo.molmass(mols_unique[i]))
        molmass.append(molinfo.molmass(mols_unique[i]))

    molmassH2=molinfo.molmass("H2")
    molmassHe=molinfo.molmass("He", db_HIT=False)
    print("H2", molmassH2, "He", molmassHe)

    return mols_unique, mols_num, molmass, molmassH2, molmassHe

def mols_num_p(mols_p, mols_unique):
    mols_num_p = []
    for k in range(len(mols_p)):
        mols_num_k = []
        for i in range(len(mols_p[k])):
            mols_num_k.append(np.where(np.array(mols_unique)==mols_p[k][i])[0][0])
        mols_num_p.append(mols_num_k)

    return mols_num_p



from exojax.spec import initspec
from exojax.spec.modit import setdgm_exomol, setdgm_hitran
def set_array(mols, db, mdb, molmass, nus, Parr, fT, T0_test, alpha_test, res_op):
    cnu=[]
    indexnu=[]
    R=[]
    pmarray=[]
    dgm_ngammaL=[]
    for i in range(len(mols)):
        cnui,indexnui,Ri,pmarrayi=initspec.init_modit(mdb[i].nu_lines,nus)
        cnu.append(cnui)
        indexnu.append(indexnui)
        R.append(Ri)
        pmarray.append(pmarrayi)

        if(db[i] == "ExoMol"):
            dgm_ngammaL.append(setdgm_exomol(mdb[i],fT,Parr,R[i],molmass[i],res_op,T0_test,alpha_test))
        elif(db[i] == "HITRAN" or db[i] == "HITEMP"):
            dgm_ngammaL.append(setdgm_hitran(mdb[i],fT,Parr,Parr*1.e-3,R[i],molmass[i],res_op,T0_test,alpha_test))

    return cnu, indexnu, R, pmarray, dgm_ngammaL



def set_array_premodit(db, mdb, nus, Ttyp=1000.):
    interval_contrast = 0.2
    dit_grid_resolution = 0.2

    lbd = []
    multi_index_uniqgrid = []
    elower_grid = []
    ngamma_ref_grid = []
    n_Texp_grid = []
    R = []
    pmarray = []
    for k in range(len(mdb)):
        lbd_k = []
        multi_index_uniqgrid_k = []
        elower_grid_k = []
        ngamma_ref_grid_k = []
        n_Texp_grid_k = []
        R_k = []
        pmarray_k = []
        for i in range(len(mdb[k])):
            if(db[k][i] == "ExoMol"):
                lbd_i, multi_index_uniqgrid_i, elower_grid_i, ngamma_ref_grid_i, n_Texp_grid_i, R_i, pmarray_i \
                    = initspec.init_premodit(
                        mdb[k][i].nu_lines,
                        nus[k],
                        mdb[k][i].elower,
                        mdb[k][i].alpha_ref,
                        mdb[k][i].n_Texp,
                        mdb[k][i].Sij0,
                        Ttyp,
                        interval_contrast=interval_contrast,
                        dit_grid_resolution=dit_grid_resolution,
                        warning=False)
            elif(db[k][i] == "HITRAN" or db[k][i] == "HITEMP"):
                mdb[k][i].n_air = np.abs(mdb[k][i].n_air)# need to check
                lbd_i, multi_index_uniqgrid_i, elower_grid_i, ngamma_ref_grid_i, n_Texp_grid_i, R_i, pmarray_i \
                    = initspec.init_premodit(
                        mdb[k][i].nu_lines,
                        nus[k],
                        mdb[k][i].elower,
                        mdb[k][i].gamma_air,
                        mdb[k][i].n_air,
                        mdb[k][i].Sij0,
                        Ttyp,
                        interval_contrast=interval_contrast,
                        dit_grid_resolution=dit_grid_resolution,
                        warning=False)

            lbd_k.append(lbd_i)
            multi_index_uniqgrid_k.append(multi_index_uniqgrid_i)
            elower_grid_k.append(elower_grid_i)
            ngamma_ref_grid_k.append(ngamma_ref_grid_i)
            n_Texp_grid_k.append(n_Texp_grid_i)
            R_k.append(R_i)
            pmarray_k.append(pmarray_i)

        lbd.append(lbd_k)
        multi_index_uniqgrid.append(multi_index_uniqgrid_k)
        elower_grid.append(elower_grid_k)
        ngamma_ref_grid.append(ngamma_ref_grid_k)
        n_Texp_grid.append(n_Texp_grid_k)
        R.append(R_k)
        pmarray.append(pmarray_k)

    return lbd, multi_index_uniqgrid, elower_grid, ngamma_ref_grid, n_Texp_grid, R, pmarray



from exojax.spec.opacalc import OpaPremodit
def set_opa_premodit(mols, mdb, nus, T_min, T_max):
    diffmode = 2
    dit_grid_resolution = 1.0
    opa = []
    for k in range(len(mdb)):
        opa_k = []
        for i in range(len(mdb[k])):
            opa_i = OpaPremodit(mdb=mdb[k][i],
                                nu_grid=nus[k],
                                diffmode=diffmode,
                                auto_trange=[T_min, T_max],
                                dit_grid_resolution=dit_grid_resolution,
                                allow_32bit=True)
            print(k, mols[k][i], np.shape(opa_i.opainfo[1])[0])

            opa_k.append(opa_i)

        opa.append(opa_k)

    return opa
