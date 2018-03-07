#! /usr/bin/env python
import os
import re
import pandas as pd
import numpy as np
import xarray as xr
import yaml

dirname = os.path.dirname(os.path.abspath(__file__))
database_file_name = os.path.join(dirname,'CMIPtools.cmip5data.db')

modeling_groups = ['BCC', 'BNU', 'CCCma', 'CMCC', 'CNRM-CERFACS', 'CSIRO-BOM',
                   'CSIRO-QCCCE', 'FIO', 'ICHEC', 'INM', 'INPE', 'IPSL',
                   'LASG-CESS', 'LASG-IAP','MIROC', 'MOHC', 'MPI-M', 'MRI',
                   'NASA-GISS', 'NCC', 'NIMR-KMA', 'NOAA-GFDL',
                   'NSF-DOE-NCAR', 'UNSW'] #'NCAR',


realms = ['atmos','land','landIce','ocean','ocnBgchem','seaIce']
frequencies = ['fx','day','mon','yr']

cmip5_root = '/glade2/collections/cmip/cmip5'


#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------
def read_var_attrs(**kwargs):
    '''query variable database for attributes dictionary
    '''
    realm = kwargs.pop('realm',None)
    varname = kwargs.pop('varname',None)
    if varname is None:
        raise ValueError('varname required')

    with open('cmip5_variables.yml','r') as f:
        cmip5_variables = yaml.load(f)

    attrs = {}
    if realm is not None:
        if varname in cmip5_variables[realm]:
            attrs = cmip5_variables[realm][varname]
    else:
        for r in cmip5_variables.keys():
            if varname in cmip5_variables[r]:
                attrs = cmip5_variables[r][varname]
    if not attrs:
        raise ValueError('"{0}" variable not found.'.format(varname))
    return attrs

#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------
def open_model_mfdataset(model,
                         experiment,
                         frequency,
                         varname,
                         realm=None,
                         ensemble=None):
    '''Open single model dataset.

    Parameters
    ----------

    model : str

    experiment : str
      i.e., 'esmHistorical'

    frequency : str

    varname : str

    realm : str,optional

    ensemble : str,optional

    Returns : xarray.Dataset
    '''

    df_subset = find_in_index(model=model,
                              experiment=experiment,
                              realm=realm,
                              frequency=frequency,
                              varname=varname,
                              ensemble=ensemble)

    #-- confirm that a dataset was returned
    if df_subset.empty:
        raise ValueError(('No files found for:\n'
                          '\tmodel = {model}\n'
                          '\texperiment = {experiment}\n'
                          '\trealm = {realm}\n'
                          '\tfrequency = {frequency}\n'
                          '\tvarname = {varname}\n'
                          '\tensemble = {ensemble}').format(model=model,
                                                            experiment=experiment,
                                                            realm=realm,
                                                            frequency=frequency,
                                                            varname=varname,
                                                            ensemble=ensemble))

    #-- realm is optional arg so check that the same varname is not in multiple realms
    realm_list = df_subset.realm.unique()
    if len(realm_list) != 1:
        raise ValueError(('"{varname}" found in multiple realms:\n'
                          '\t{realm_list}\n'
                          'Specify the realm to use').format(realm_list=realm_list,
                                                             varname=varname))

    #-- make a list of ensemble members, read a concat dataset along ens dim
    ensemble_list = df_subset.ensemble.unique()
    ds_list = []
    for ens_i in ensemble_list:
        ens_match = (df_subset.ensemble == ens_i)
        files = df_subset.loc[ens_match].files.tolist()
        ds_list.append(xr.open_mfdataset(files))

    return xr.concat(ds_list,dim='ens')

#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------
def find_files_in_index(**kwargs):
    '''return files according to requested data.

    '''
    df_subset = find_in_index(**kwargs)

    return sorted(df_subset.files.tolist())

#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------
def find_in_index(**kwargs):
    '''return subset of database according to requested data.

    '''

    spec = {'model' : kwargs.pop('model',None),
            'experiment' : kwargs.pop('experiment',None),
            'realm' : kwargs.pop('realm',None),
            'frequency' : kwargs.pop('frequency',None),
            'ensemble' : kwargs.pop('ensemble',None),
            'varname' : kwargs.pop('varname',None)}

    df = pd.read_pickle(database_file_name)

    query = np.ones(len(df),dtype=bool)
    for key,val in spec.items():
        if val is not None:
            query = query & (df[key] == val)

    return df.loc[query]

#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------
def make_index():
    ''' write database for all the data on disk.
    '''
    df = pd.DataFrame()

    vYYYYMMDD = re.compile(r'v\d{4}\d{2}\d{2}')
    vN = re.compile(r'v\d{1}')
    ens_patt = re.compile(r'r\di\dp\d')

    #walk = [os.walk(os.path.join(cmip5_root,o,g))
    #        for o in os.listdir(cmip5_root) for g in modeling_groups]

    #for w in walk:
    for o in os.listdir(cmip5_root):
        for g in modeling_groups:

            print(os.path.join(cmip5_root,o,g))
            w = os.walk(os.path.join(cmip5_root,o,g))

            for root, dirs, files in w:

                #-- we want to get to the files
                if not files: continue

                #-- avoid instances of other files, some .log files found
                sfiles = sorted([f for f in files if f.endswith('.nc')])
                if not sfiles: continue

                #-- parse the path to this location
                root_split = root.split('/')

                #-- Handle path anomalies
                # root = path-to-here/experiment/freq/realm/realm-freq/ensemble/version
                if vYYYYMMDD.findall(root_split[-1]):
                    version = root_split[-1]
                    realm = root_split[-4]
                    frequency = root_split[-5]
                    experiment = root_split[-6]
                    model = root_split[-7]

                # root = path-to-here/experiment/freq/realm/realm-freq/ensemble/version/varname
                elif vYYYYMMDD.findall(root_split[-2]):

                    version = root_split[-2]
                    realm = root_split[-5]
                    frequency =  root_split[-6]
                    experiment = root_split[-7]
                    model = root_split[-8]

                # root = path-to-here/experiment/freq/realm/realm-freq/ensemble/varname_version
                elif re.findall(r'\w+_\d{4}\d{2}\d{2}',root_split[-1]):
                    version = 'v'+re.findall(r'\w+_\d{4}\d{2}\d{2}',
                                             root_split[-1])[0].split('_')[-1]
                    realm = root_split[-5]
                    frequency =  root_split[-6]
                    experiment = root_split[-7]
                    model = root_split[-8]

                # root = path-to-here/experiment/freq/realm/realm-freq/ensemble/v{1,2}
                elif vN.findall(root_split[-1]):
                    version = root_split[-1]
                    realm = root_split[-4]
                    frequency = root_split[-5]
                    experiment = root_split[-6]
                    model = root_split[-7]

                # root = path-to-here/experiment/freq/realm/realm-freq/ensemble/v{1,2}/varname
                elif vN.findall(root_split[-2]):
                    version = root_split[-2]
                    realm = root_split[-5]
                    frequency = root_split[-6]
                    experiment = root_split[-7]
                    model = root_split[-8]

                # root = path-to-here/experiment/freq/realm/realm-freq/ensemble
                elif ens_patt.findall(root_split[-1]):
                    version = 'v0'
                    realm = root_split[-3]
                    frequency = root_split[-4]
                    experiment = root_split[-5]
                    model = root_split[-6]

                else:
                    print('I am lost: {0}'.format(root))
                    continue

                print(('\tadding {n} files for '
                       '{model} {experiment} {realm} '
                       '{frequency}').format(n=len(sfiles),
                                             model=model,
                                             experiment=experiment,
                                             realm=realm,
                                             frequency=frequency))

                db_entry = []
                for f in sfiles:
                    entry = {'version' : version,
                             'realm' : realm,
                             'frequency' : frequency,
                             'file_basename' : f,
                             'files' : os.path.join(root,f)}

                    fsplit = f.split('_')
                    for i,k in enumerate(['varname','realm_freq','model','experiment','ensemble']):
                        try:
                            entry[k] = fsplit[i]
                        except:
                            print('split failed: {0}'.format(f))

                    db_entry.append(entry)

                df = df.append(db_entry,ignore_index=False)


    #-- cull duplicates: eliminate all but latest version
    df = df.reset_index()
    df = df.sort_values('version').drop_duplicates(subset='file_basename',keep='last')
    df = df.reset_index()

    print('\n')
    df.info()
    print('writing to {0}'.format(database_file_name))
    df.to_pickle(database_file_name)
    print('done.')

if __name__ == '__main__':
    #--- setup index
    make_index()
