#!/opt/local/bin/python

import numpy as ny
import scipy.fftpack as fft

class TimingAnalysis(object):
    """ Timing Analysis. """
    def __init__(self,verbose=True):
        pass

    def t_fold(self,data,period,nbin=-1):
        """ Take data of shape (2,N) containing time,signal values for N samples 
             and fold with given period in nbin bins.
             period should have same dimensions as data[0].
             Use nbin < 0 for equi-spaced data.
             Output array of shape (2,nbin) or (2,dN) containing phase, average signal values.
        """
        if len(data.shape) != 2:
            raise TypeError("Data should have shape (2,N). Detected ("+','.join([str(i) for i in data.shape])+").")
        if data.shape[0] != 2:
            raise TypeError("Data should have shape (2,N). Detected ("+','.join([str(i) for i in data.shape])+").")
        N = data.shape[1]
        N_chunks = ny.floor(data[0].max()/period).astype(int)
        dN = N/N_chunks
        N_rem = N%N_chunks
        data_trim = ny.delete(data,ny.s_[-N_rem:],axis=1) if N_rem > 0 else data.copy()

        nb = nbin if nbin > 0 else dN
        folded = ny.zeros((2,nb),dtype=float)        
        phasebin = ny.linspace(0.0,1.0,nb+1)
        folded[0] = 0.5*(phasebin[1:]+phasebin[:-1]) if nbin > 0 else 1.0*ny.arange(dN)/dN
        
        dnb = dN/(nb-1)
        nb_rem = dN%(nb-1) # first nb-1 bins (in each chunk) will get dnb samples each, last will get nb_rem samples
        dest_bin = ny.zeros(data_trim.shape[1],dtype=int)
        for c in range(N_chunks):
            for b in range(nb-1):
                # print 'hi',b*dnb+c*dN,(b+1)*dnb+c*dN
                dest_bin[b*dnb+c*dN:(b+1)*dnb+c*dN] = b
            # print 'HO', (nb-1)*dnb+c*dN,(nb-1)*dnb+nb_rem+c*dN
            dest_bin[(nb-1)*dnb+c*dN:(nb-1)*dnb+nb_rem+c*dN] = nb-1

        for b in range(nb):
            folded[1,b] = ny.sum(data_trim[1][dest_bin == b])

        return folded


    def optimal_period(self,data,N_T=100,nbin=10):
        """ Use least squares folding to find optimal period. in data of shape (2,N)."""
        if len(data.shape) != 2:
            raise TypeError("Data should have shape (2,N). Detected ("+','.join([str(i) for i in data.shape])+").")
        if data.shape[0] != 2:
            raise TypeError("Data should have shape (2,N). Detected ("+','.join([str(i) for i in data.shape])+").")

        Tmin = (data[0,1]-data[0,0])*50
        Tmax = data[0].max()*0.25
        Tvals = ny.logspace(ny.log10(Tmin),ny.log10(Tmax),N_T)
        chi2 = ny.zeros(N_T,dtype=float)
        for t in range(N_T):
            folded = self.t_fold(data,Tvals[t],nbin=nbin)
            chi2[t] = ny.sum((folded[1] - folded[1].mean())**2)
        t_opt = ny.argmax(chi2)

        return Tvals[t_opt],chi2,Tvals

    def fourier(self,data,nbin=100):
        time_series = data[1]-data[1].mean()
        tbins = ny.linspace(data[0,0],data[0,-1],nbin)
        tgrid = 0.5*(tbins[1:]+tbins[:-1])
        time_binned,bins = ny.histogram(data[0],bins=tbins,weights=data[1],density=False)
        norm,bins = ny.histogram(data[0],bins=tbins,density=False)
        time_binned = 1.0*time_binned/(norm + 1e-15)
        ft = fft.fft(time_binned)
        ft_nu = 1.0/tbins[-1]*ny.arange(tgrid.size)
        return ft,ft_nu

    # def t_fold_alt(self,data,period,nbin=-1):
    #     """ Take data of shape (2,N) containing time,signal values for N samples 
    #          and fold with given period in nbin bins.
    #          period should have same dimensions as data[0].
    #          Use nbin < 0 for equi-spaced data.
    #          Output array of shape (2,nbin) or (2,dN) containing phase, average signal values.
    #     """
    #     if len(data.shape) != 2:
    #         raise TypeError("Data should have shape (2,N). Detected ("+','.join([str(i) for i in data.shape])+").")
    #     if data.shape[0] != 2:
    #         raise TypeError("Data should have shape (2,N). Detected ("+','.join([str(i) for i in data.shape])+").")
    #     N = data.shape[1]
    #     N_chunks = ny.floor(data[0].max()/period).astype(int)
    #     dN = N/N_chunks
    #     N_rem = N%N_chunks
    #     data_trim = ny.delete(data,ny.s_[-N_rem:],axis=1) if N_rem > 0 else data.copy()
    #     data_trim = ny.reshape(data_trim,(2,N_chunks,dN))

    #     nb = nbin if nbin > 0 else dN
    #     folded = ny.zeros((2,nb),dtype=float)        
    #     phasebin = ny.linspace(0.0,1.0,nb+1)
    #     folded[0] = 0.5*(phasebin[1:]+phasebin[:-1]) if nbin > 0 else 1.0*ny.arange(dN)/dN
    #     if nbin > 0:
    #         for c in range(N_chunks):
    #             tmax,tmin = data_trim[0,c,-1],data_trim[0,c,0]
    #             phasevals = (data_trim[0,c]-tmin)/(tmax-tmin)
    #             freq,bins = ny.histogram(phasevals,bins=phasebin,weights=data_trim[1,c],density=False)
    #             # norm,bins = ny.histogram(phasevals,bins=phasebin,density=False)
    #             norm = 1.0*dN/nbin
    #             folded[1] += freq/(norm + 1e-15)
    #         folded[1] /= N_chunks
    #     else:
    #         folded[1] = ny.mean(data_trim[1],axis=0)
    #     return folded

if __name__=="__main__":
    import matplotlib.pyplot as plt
    ta = TimingAnalysis()
    infile = "../../data/CMA_F_as1_lc1.txt"
    data = ny.loadtxt(infile).T

    # folded = ta.t_fold_alt(data,10.0,nbin=10)

    # plt.plot(data[0],data[1],lw=0.25)
    # plt.show()

    # ft,nu = ta.fourier(data,nbin=data.shape[1])
    # plt.xlim([0,100])
    # plt.plot(nu,2*ny.absolute(ft)**2/ft[0].real)
    # plt.savefig('../../plots/powspec.png')

    NBIN = -40
    N_T=1600
    T_opt,chi2,Tvals = ta.optimal_period(data,N_T=N_T,nbin=NBIN)
    print "Optimal period = {0:.5f} s".format(T_opt)
    folded_opt = ta.t_fold(data,T_opt,nbin=NBIN)

    plt.xscale('log')
    plt.yscale('log')
    plt.plot(Tvals,chi2)
    plt.savefig('../../plots/folding_chi2.png')
    plt.close()

    T1 = 0.62
    folded = ta.t_fold(data,T1,nbin=NBIN)

    plt.plot(folded_opt[0],folded_opt[1],'-',c='b',label='optimal')
    plt.plot(folded[0],folded[1],'--',c='k',label='fixed')
    plt.legend()
    plt.savefig('../../plots/folding_optimal.png')
    plt.close()
