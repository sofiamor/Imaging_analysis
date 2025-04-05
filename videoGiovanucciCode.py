import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from pynwb import NWBFile, NWBHDF5IO
from pynwb.image import ImageSeries

try:
    cv2.setNumThreads( )
except:
    pass

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except:
    pass

from datetime import datetime
from dateutil.tz import tzlocal

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.paths import caiman_datadir

# Set up the logger (optional); change this if you like.
# You can log to a file using the filename parameter, or make the output more
# or less verbose by setting level to logging.DEBUG, logging.INFO,
# logging.WARNING, or logging.ERROR

logging.basicConfig(format=
"%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
"[%(process)d] %(message)s",
level=logging.WARNING)

def main():
    pass # For compatibility between running under Spyder and the CLI

    # Select file(s) to be processed (download if not present)
    fnames = [os.path.join(caiman_datadir(), 'F:/histed/TESThisted2101_200ms_tstack.nwb')]
    # estimates save path can be same or different from raw data path
    save_path = os.path.join(caiman_datadir(), 'F:/histed/TESThisted2101_200ms_tstack.nwb')
    # filename to be created or processed
    # dataset dependent parameters
    fr = 15 # Imaging rate in frames per second
    decay_time = 0.4 # Length of a typical transient in seconds

    #%% load the file and save it in the NWB format (if it doesn't exist already)
    fnames_orig = 'histed_2101_test2_n2_200msSTIM_MMStack.ome.tif' # filename to be processed
    if not os.path.exists(fnames[0]):
        frames_orig = 'histed_2101_test2_n2_200msSTIM_MMStack.ome.tif' # filename to be processed
        if frames_orig in ['histed_2101_test2_n2_200msSTIM_MMStack.ome.tif']:
            frames_orig = [download_demo(fnames_orig)]
            print(f"Downloading {frames_orig}...")
        orig_movie = cm.load(fnames_orig, fr=fr)

    # save file in NWB format with various additional info
    orig_movie.save(fnames[0], sess_desc='test', identifier='demo 1',
        imaging_plane_description='single plane',
        emission_lambda = 520.0, 
        indicator='GCAMP6f',
        location='Superior Colliculus',
        device_name='OnePhotonMicroscope',
        experimenter='Sofia Morou', lab_name='Kardamakis Lab',
        institution='Instituto de Neurociencias',
        experiment_description='optostimulating and Ca2+ imaging of histed virus',
        session_id='Session 1',
        var_name_hdf5='TwoPhotonSeries')

    #%% First setup some parameters for data and motion correction

    #
    # motion correction parameters
    dxy = (2., 2.) # spatial resolution in x and y in (um per pixel)
    # note the lower than usual spatial resolution here
    max_shift_um = (12., 12.) # maximum shift in um
    patch_motion_um = (100, 100) # patch size for non-rigid correction in um
    pw_rigid = True # flag to select rigid vs pw_rigid motion correction
    # maximum allowed rigid shift in pixels
    max_shifts = [int(a/b) for a, b in zip(max_shift_um, dxy)]
    # start a new patch for pw-rigid motion correction every x pixels
    strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
    # overlap between patches (size of patch in pixels: strides+overlaps)
    overlaps = (24, 24)
    # maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3

    print(f"File exists: {os.path.exists(fnames)}")
    print(f"File type: {fnames.split('.')[-1]}")

    mc_dict = {
        'data': {
            'fnames': fnames,  # Ensure this is a valid filename or list
            'fr': fr,
        },
        'motion': {
            'strides': strides,  # Example value, adjust based on your setup
            'overlaps': overlaps,  # Example value, adjust based on your setup
            'max_deviation_rigid': max_deviation_rigid,  # Maximum deviation for motion correction
        },
        'init': {
            'method_init': 'corr_pnr',  # Initialization method
            'min_corr': 0.8,
            'min_pnr': 10,
        },
        'spatial': {
            'gSig': (3, 3),  # Spatial filter size
            'gSiz': (13, 13),
        },
        'temporal': {
            'p': 1,  # Order of autoregressive model
        },
        'merging': {
            'merge_thr': 0.85,  # Merge threshold
        },
    }

    opts = params.CNMFParams(params_dict=mc_dict)


    # XX play the movie (optional)
    # playing the movie using opencv. It requires loading the movie in memory.
    # To close the video press q
    display_images = True
    if display_images:
        m_orig = cm.load_movie_chain(fnames, var_name_hdf5=opts.data['var_name_hdf5'])
        ds_ratio = 0.2
        movichandle = m_orig.resize(1, 1, ds_ratio)
        movichandle.play(q_max=99.5, fr=60, magnification=2)

    # XX start a cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

    # XX MOTION CORRECTION
    # First we create a motion correction object with the specified parameters
    mc = MotionCorrect(fnames, dview=dview, var_name_hdf5=opts.data['var_name_hdf5'], **opts.get_group('motion'))
    # note that the file is not loaded in memory

    # XX Run (piecewise-rigid motion) correction using NoRMCorre
    mc.motion_correct(save_movie=True)

    # XX compare with original movie
    if display_images:
        m_orig = cm.load_movie_chain(fnames, var_name_hdf5=opts.data['var_name_hdf5'])
        m_els = cm.load(mc.mmap_file)
        ds_ratio = 0.2
        movichandle = cm.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.shifts_movie*mc.nonneg_movie,
                                    m_els.resize(1, 1, ds_ratio)], axis=2)
        movichandle.play(fr=60, q_max=99.5, magnification=2) # press q to exit

    # XX MEMORY MAPPING
    border_to_0 = 0 if mc.border_nan is 'copy' else mc.border_to_0
    # you can include the boundaries of the FOV if you used the 'copy' option
    # during motion correction, although be careful about the components near
    # the boundaries

    # memory map the file in order 'C'
    fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap', order='C',
                            border_to_0=border_to_0) # exclude borders

    # now load the file
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # load frames in python format (T x X x Y)

    # XX restart cluster to clean up memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

    # now load the file
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # load frames in python format (T x X x Y)

    # XX restart cluster to clean up memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

    # XX parameters for source extraction and deconvolution
    p = 1 # order of the autoregressive system
    gnb = 2 # number of global background components
    merge_thr = 0.85  # merging threshold, max correlation allowed
    rf = 11 # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 4 # amount of overlap between the patches in pixels
    K = 1 # number of components per patch
    gSig = [4 , 4]  # expected half size of neurons in pixels
    # initialization method (if analyzing dendritic data using 'sparse_nmf')
    method_init = 'greedy_roi'
    ssub = 4 # spatial subsampling during initialization
    tsub = 1 # temporal subsampling during initialization

    # parameters for component evaluation
    opts_dict = {'fnames': fnames,
        'fr': fr,
        'nb': gnb,
        'rf': rf,
        'K': K,
        'gSig': gSig,
        'stride': stride_cnmf,
        'method_init': method_init,
        'rolling_sum': True,
        'merge_thr': merge_thr,
        'n_processes': n_processes,
        'only_init': True,
        'ssub': ssub,
        'tsub': tsub}

    opts.change_params(params_dict=opts_dict)

    # XX RUN CNMF ON PATCHES
    # First extract spatial and temporal components on patches and combine them
    # for this step deconvolution is turned off (p=0)
    opts.change_params({'p': 0})
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)

    # XX ALTERNATE WAY TO RUN THE PIPELINE AT ONCE
    # you can also perform the motion correction plus cnmf fitting steps
    # simultaneously after defining your parameters object using
    # cnm1 = cnmf.CNMF(n_processes, params=opts, dview=dview)
    # cnm1.fit_file(motion_correct=True)

    # XX plot contours of found components
    Cn = cm.local_correlations(images, swap_dim=False)
    Cn[np.isnan(Cn)] = 0
    cnm.estimates.plot_contours(img=Cn)
    plt.title('Contour plots of found components')

    # XX save results in a separate file (just for demonstration purposes)
    cnm.estimates.Cn = Cn
    cnm.save(fname_new[:-4]+'hdf5')
    # cm.movie(Cn).save(fname_new[:-5]+'.Cn.tif')

    # XX RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
    cnm.params.change_params({'p': p})
    cnm2 = cnm.refit(images, dview=dview)

    # XX COMPONENT EVALUATION
    # the components are evaluated in three ways:
    # a) the shape of each component must be correlated with the data
    # b) a minimum peak SNR is required over the length of a transient
    # c) each shape passes a CNN based classifier
    min_SNR = 1  # signal to noise ratio for accepting a component
    rval_thr = 0.85  # space correlation threshold for accepting a component
    cnn_thr = 0.99  # threshold for CNN based classifier
    cnn_lowest = 0.1  # neurons with cnn probability lower than this value are rejected

    cnm2.params.set('quality', {'decay_time': decay_time,
        'min_SNR': min_SNR,
        'rval_thr': rval_thr,
        'use_cnn': True,
        'min_cnn_thr': cnn_thr,
        'cnn_lowest': cnn_lowest})
    cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)

    cnm2.estimates.Cn = Cn
    cnm2.save(fname_new[:-4] + 'hdf5')

    # XX PLOT COMPONENTS
    cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)

    # XX VIEW TRACES (accepted and rejected)
    if display_images:
        cnm2.estimates.view_components(images, img=Cn,
            idx=cnm2.estimates.idx_components)
        cnm2.estimates.view_components(images, img=Cn,
            idx=cnm2.estimates.idx_components_bad)

    # XX update object with selected components
    # cnm2.estimates.select_components(use_object=True)

    # XX Extract DF/F values
    cnm2.estimates.detrend_df_f(quantileMin=1, frames_window=50)
    cnm2.estimates.view_components(img=Cn)

    # XX reconstruct denoised movie (press q to exit)
    if display_images:
        cnm2.estimates.play_movie(images, q_max=99.9, gain_res=2,
            magnification=2,
            border_to_0=border_to_0,
            include_bck=False) # background not shown

    # XX STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)

    # XX save the results in the original NWB file
    cnm2.estimates.save_NWB(save_path, imaging_rate=fr, session_start_time=datetime.now(tzlocal()),
        raw_data_file=fnames[0])