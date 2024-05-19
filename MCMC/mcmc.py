import numpy as np
import sys,os
from mpi4py import MPI

from cobaya.run import run
from cobaya.log import LoggedError

# below three can be command line args
Parallel = True
Resume = False # useful for restarting stopped jobs
id_str = 'strtline' # label for chains (and plots)

# set up folder locations
Base_Dir = '/home/aseem/iucaa/Workshops-Seminars/HPC-Srinagar-May2024/share/hpctutorial/MCMC/'
Like_Dir = Base_Dir + 'likes/'
Data_Dir = Base_Dir + 'data/'
Plots_Dir = Base_Dir + 'plots/'

# load data
data_file = Data_Dir + 'data.txt'
data_prods = np.loadtxt(data_file).T # shape (3,ndata)

# set up initialization arrays
xvals = data_prods[0]
data = data_prods[1]
sigma = data_prods[2]
invcov_mat = np.diagflat(1/sigma**2) # change according to use case

# setup runtime args
Rminus1_Stop = 0.01 
Rminus1_CL_Stop = 0.05 # 0.05
Rminus1_CL_Level = 0.95 # 95

# setup info dictionary
info = {}
# -- likelihood
info['likelihood'] = {'likelihoods.Chi2Like':
                      {'python_path':Like_Dir,
                       'data':data,'invcov_mat':invcov_mat}}

# -- theory
info['theory'] = {'likelihoods.StraightLineTheory':
                  {'python_path':Like_Dir,
                   'xvals':xvals}}

# -- parameters and sampling
info['params'] = {}

# # basic structure
# info['params'][par_string] = {'ref':{distribution},
#                               'prior':{distribution},
#                               'proposal':value,
#                               'latex':latex_string}

# note ordering of parameters, consistent with likelihood code
info['params']['a0'] = {'ref':{'min':-0.01,'max':0.01},
                        'prior':{'min':-10,'max':10},
                        'proposal':0.01,
                        'latex':'a_{0}'}

info['params']['a1'] = {'ref':{'min':-0.01,'max':0.01},
                        'prior':{'min':-10,'max':10},
                        'proposal':0.01,
                        'latex':'a_{1}'}

# -- sampler
info['sampler'] = {'mcmc':
                   {'learn_proposal': True,
                    'measure_speeds': True,
                    'max_samples': 10000000,
                    'max_tries': 1000,
                    'Rminus1_stop': Rminus1_Stop,
                    'Rminus1_cl_stop': Rminus1_CL_Stop,
                    'Rminus1_cl_level': Rminus1_CL_Level,
                    'burn_in': 0}}

# -- output location
info['output'] = 'stats/chains/' + id_str

# -- resume stopped job or start new job and overwrite
if Resume:
    info['resume'] = True
else:
    info['force'] = True

# main run commands
if Parallel:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    success = False
    try:
        upd_info, sampler = run(info,resume=Resume) # note second appearance of Resume
        success = True
    except LoggedError as err:
        pass
    success = all(comm.allgather(success))
    if not success and rank == 0:
        print("Sampling failed!")

    print('all done on processor rank {0:d}'.format(rank))
else:
    upd_info, sampler = run(info,resume=Resume) # note second appearance of Resume
    print('all done')


# analyse and plot
if rank == 0:
    import scipy.special as sysp
    import matplotlib.pyplot as plt
    import matplotlib.colors as pltcol

    from getdist.mcsamples import loadMCSamples
    import getdist.plots as gdplt
    import gc

    Save_Fig = True
    Burn_Frac = 0.3
    rng = np.random.RandomState(42)
    dim = 2
    dof = xvals.size - dim

    gd_sample = loadMCSamples(os.path.abspath(info["output"]),settings={'ignore_rows':Burn_Frac})
    gd_sample.label = 'MCMC'
    # samples contain params | chi2 | chi2__name | ?? | ??
    mcmc_covmat = gd_sample.getCovMat().matrix[:dim, :dim]

    sample = gd_sample.samples
        
    sample = sample.T
    ibest = sample[-2].argmin()
    mcmc_best = sample[:dim,ibest]
    mcmc_chi2 = sample[-2,ibest]
    pval = sysp.gammainc(mcmc_chi2/2,dof/2)

    mcmc_sig = np.sqrt(np.diag(mcmc_covmat))

    print('MCMC...')
    print("... best fit ( a0,a{0:d}) = ( ".format(dim)+','.join(['%.4e' % (pval,) for pval in mcmc_best])+" )")
    print("... std dev  ( a0,a{0:d}) = ( ".format(dim)+','.join(['%.4e' % (pval,) for pval in mcmc_sig])+" )")
    print("... chi2_best,dof,chi2_red,pval: {0:.3f},{1:d},{2:.3f},{3:.3e}".format(mcmc_chi2,dof,mcmc_chi2/dof,pval))

    plot_param_list = ['a0','a1']
    Subplot_Size = 1.6 

    gdplot = gdplt.get_subplot_plotter(subplot_size=Subplot_Size)
    gdplot.settings.num_plot_contours = 3
    # gdplot.settings.axes_fontsize = FS3
    # gdplot.settings.axes_labelsize = FS2
    # gdplot.settings.title_limit_fontsize = FS2

    gdplot.triangle_plot([gd_sample], plot_param_list,
                         filled=[True],contour_colors=['indigo'],legend_loc='upper center',
                         markers=mcmc_best,
                         marker_args={'c': 'indigo','ls': '--','lw': 1.5,'alpha': 0.6},
                         title_limit=1)
    for par_y in range(dim):
        str_y = plot_param_list[par_y]
        ax = gdplot.subplots[par_y,par_y]
        ax.axvline(mcmc_best[par_y],c='indigo',ls='--',lw=1.5,alpha=0.6)
        for par_x in range(par_y):
            str_x = plot_param_list[par_x]
            ax = gdplot.subplots[par_y,par_x]
            ax.scatter([mcmc_best[par_x]],[mcmc_best[par_y]],
                       marker='*',s=100,c='white')

    if Save_Fig:
        filename = 'contours_' + id_str + '.png'
        print('Writing to file: '+Plots_Dir+filename)
        gdplot.export(fname=filename,adir=Plots_Dir)

    
    a0_best,a1_best = mcmc_best[:dim]
    model_best = a0_best + xvals*a1_best

    N_Boot_Cobaya = np.min([1000,int(0.2*sample[0].size)])
    Ind = gd_sample.random_single_samples_indices(random_state=42,max_samples=N_Boot_Cobaya)
    N_Boot_Cobaya = Ind.size
    print('N_Boot_Cobaya: ',N_Boot_Cobaya)

    model_boot = np.zeros((N_Boot_Cobaya,xvals.size),dtype=float)

    print('... extracting stats from subsample')
    for b in range(N_Boot_Cobaya):
        a0_b,a1_b = sample[:dim,Ind[b]] 
        model_b = a0_b + xvals*a1_b
        model_boot[b] = model_b

    model_16pc = np.percentile(model_boot,16,axis=0)
    model_84pc = np.percentile(model_boot,84,axis=0)

    del model_boot
    gc.collect()

    cols = ['crimson','indigo','forestgreen']
    plt.figure(figsize=(7,7))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.plot(xvals,model_best,'-',lw=1,c=cols[0],label='best fit')
    plt.fill_between(xvals,model_84pc,model_16pc,color=cols[0],alpha=0.15)

    plt.errorbar(xvals,data,yerr=sigma,c=cols[0],ls='none',capsize=5,marker='o',markersize=4,label='data')
    
    plt.legend(loc='upper right')
    plt.minorticks_on()
    if Save_Fig:
        filename = Plots_Dir+'stats_' + id_str + '.png'
        print('Writing to file: '+filename)
        plt.savefig(filename,bbox_inches='tight')
    plt.show()
