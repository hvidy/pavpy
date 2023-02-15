import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import scipy.optimize as optimization
from scipy.io import readsav

from .utils import (
    get_uv,
    get_diams,
    get_ldcs,
    wtmn,
    randomcorr
)
from .models import (
    ud,
    ld,
    ldd_claret4,
    ellipse
)

class PavoObs():
    """
    PavoObs objects provide user-friendly attribute access and functions
    for calibrating pavo data and model fitting

    Parameters
    ----------


    Attributes
    ----------


    Notes
    -----


    Examples
    --------

    """

    def __init__(self, filename, **kwargs):

        self.df = pd.read_fwf(filename, skiprows=1, names=['Star','Scan','JD','CHARA_num','wl','v2',
            'v2c','v2sig','v2exp','flux','T1','T2','File'], infer_nrows=2000)

        self.filenames = self.df.File.unique()

        self.baselines = get_uv(self.df)

        self.df = self.df.assign(sp_freq = self.baselines.loc[self.df.File].bl.values / self.df.wl,
                                 sp_freq_sample = self.baselines.loc[self.df.File].bl.values / self.df.wl,
                                 pa = self.baselines.loc[self.df.File].pa.values)

        self.caldiams = get_diams(self.df, 0.05)

        self.datadir = '/'.join(filename.split('/')[:-1])+'/'


    def t0corr(self,scan):

        ind = self.df.File == scan

        x = np.arange(0,8,0.01)
        v2_mod = np.sqrt((1 + 0.387*x**2)/(1 + 3.04*x**2 + 2.21*x**4))
        rat_mod = np.sqrt((1 + 0.322*x**2)/(1 + 9.5*x**2 + 49.8*x**4))

        rat = self.df[ind].corr_v2c/self.df[ind].corr_v2
        good = self.df[ind].corr_v2sig/self.df[ind].corr_v2 > 1e-5
        xdata = np.interp(rat,np.flip(rat_mod),np.flip(x))

        x1mu = wtmn(xdata[good]*self.df[ind].wl[good], self.df[ind].corr_v2sig[good]/self.df[ind].corr_v2[good])

        self.df.loc[ind,'corr_v2'] = self.df[ind].corr_v2 / np.interp(x1mu/self.df[ind].wl, x, v2_mod)


    def calc_sysv2(self, calscans):
        # Calculate system visibility from a set of calibrator scans
        
        weighted_sum = pd.Series(dtype='float64')
        sum_weights = pd.Series(dtype='float64')
        
        for calscan in calscans:
            scan = self.df[self.df.File==calscan].reset_index()
            star = scan['Star'][0]
        
            #Expected visibility for a star of a particular diameter at observed spatial frequency
            exp_v2 = ud(self.df.sp_freq_sample, self.caldiams.loc[star].sample_diameter)
        
            #System visibility = observed visibility/expected visibility
            sysv2 = scan['corr_v2']/exp_v2
            sysv2e = scan['corr_v2sig']/exp_v2
            
            weighted_sum = weighted_sum.add(sysv2*sysv2e**-2,fill_value=0)
            sum_weights = sum_weights.add(sysv2e**-2,fill_value=0)
        
        meansysv2 = weighted_sum/sum_weights

        if len(calscans) > 1:
            mse = pd.Series(dtype='float64')

            for calscan in calscans:
                scan = self.df[self.df.File==calscan].reset_index()
                star = scan['Star'][0]

                #Expected visibility for a star of a particular diameter at observed spatial frequency
                exp_v2 = ud(self.df.sp_freq_sample, self.caldiams.loc[star].sample_diameter)

                #System visibility = observed visibility/expected visibility
                sysv2 = scan['corr_v2']/exp_v2
                sysv2e = scan['corr_v2sig']/exp_v2

                mse = mse.add((sysv2 - meansysv2)**2*sysv2e**-2, fill_value=0)

            mse = mse/(len(calscans)-1)
            mse[mse < 1] = 1

        else:
            mse = 1

        meansysv2e = ((2 + (len(calscans)-1)*mse)/((len(calscans)+1)*sum_weights))**0.5
        
        return meansysv2, meansysv2e


    def calibrate(self, config, scale = None):

        self.calibrated = pd.DataFrame()

        self.df = self.df.assign(corr_v2 = self.df.v2, corr_v2c = self.df.v2c, corr_v2sig = self.df.v2sig)

        for row in config.itertuples():
            
            # Apply corrections
            scans = row.calscans.copy()
            scans.append(row.targetscans)

            if row.exp is True:
                for scan in scans:
                    ind = self.df.File == scan
                    self.df.loc[ind,'corr_v2'] = self.df[ind].v2/self.df[ind].v2exp
                    self.df.loc[ind,'corr_v2c'] = self.df[ind].v2c/self.df[ind].v2exp
                    self.df.loc[ind,'corr_v2sig'] = self.df[ind].v2sig/self.df[ind].v2exp
            else:
                for scan in scans:
                    # Need to reset corrected visibilities in case they were set by a previous bracket
                    ind = self.df.File == scan
                    self.df.loc[ind,'corr_v2'] = self.df[ind].v2
                    self.df.loc[ind,'corr_v2c'] = self.df[ind].v2c
                    self.df.loc[ind,'corr_v2sig'] = self.df[ind].v2sig

            if row.t0 is True:
                for scan in scans:
                    self.t0corr(scan)

            if scale is not None:
                #Visibilities perturbed by covariance matrix
                for scan in scans:
                    ind = self.df.File == scan
                    sav_data = readsav(self.datadir+scan+'.cov')
                    cov_mat = np.squeeze(sav_data.covmatres)
                    noise = randomcorr(cov_mat*scale)
                    self.df.loc[ind,'corr_v2'] = self.df.loc[ind,'corr_v2'] + noise

            sys_v2, sys_v2sig = self.calc_sysv2(row.calscans)

            result = self.df[self.df.File == row.targetscans].reset_index()
            result = result.assign(cal_v2 = result.corr_v2 / sys_v2)
            result = result.assign(cal_v2sig = np.abs(result.cal_v2*((result.corr_v2sig/result.corr_v2)**2 + (sys_v2sig/sys_v2)**2)**0.5))

            result = result.query('wl > '+str(row.wl_min)+' and wl < '+str(row.wl_max))

            self.calibrated = pd.concat([self.calibrated,result])

    def fit_model(self, model=ud, p0=np.array([1.0]), fixed=[False], individual=False):

        if len(fixed) > 1:
            assert (len(fixed) == len(p0)), "Ambiguous which parameters are to be fixed: p0 and fixed must have same length"
            assert (len(fixed) > np.sum(fixed)), "At least one parameter must not be fixed"
        else:
            assert (fixed[0] is False), "At least one parameter must not be fixed"

        if type(model) is str:
            if model == 'ud':
                model = ud
            elif model == 'ld':
                model = ld
            else:
                try:
                    exec('from .models import '+model)
                    model = vars()[model]
                except ImportError as err:
                    print("Function "+model+" is not implemented in the models module")
                    raise

        if np.sum(fixed) > 0:
            parameters = 'x'
            arguments = 'x'
            for idx,p in enumerate(p0):
                if fixed[idx] is True:
                    arguments = arguments + ', '+str(p)
                else:
                    # This will be a variable of the lambda function
                    parameters = parameters + ', a'+str(idx)
                    arguments = arguments + ', a'+str(idx)

            func = eval("lambda "+parameters+": model("+arguments+")",{'model':model})

            p0 = p0[~np.asarray(fixed)]
        else:
            func = model

        if individual is False:
            
            popt, pcov  = optimization.curve_fit(func, 
                                                 self.calibrated.sp_freq, 
                                                 self.calibrated.cal_v2, 
                                                 p0 = p0, 
                                                 sigma = self.calibrated.cal_v2sig)

            self.fit = [{'model': func, 'parameters': popt, 'covariance': pcov}]

        else:

            fits = []

            for filename in self.calibrated.File.unique():
                scan = self.calibrated[self.calibrated.File==filename].reset_index()
                popt, pcov  = optimization.curve_fit(func, 
                                                     scan.sp_freq, 
                                                     scan.cal_v2, 
                                                     p0 = p0, 
                                                     sigma = scan.cal_v2sig)
                fits.append({'model': func, 'parameters': popt, 'covariance': pcov, 'bracket': filename, 'pa': scan.pa[0]})

            self.fit = fits

    def fit_ldmodel(self, ld_coeffs, model=ldd_claret4, p0=np.array([1.0])):

        a1 = np.tile(ld_coeffs.a1, self.calibrated.File.unique().shape)
        a2 = np.tile(ld_coeffs.a2, self.calibrated.File.unique().shape)
        a3 = np.tile(ld_coeffs.a3, self.calibrated.File.unique().shape)
        a4 = np.tile(ld_coeffs.a4, self.calibrated.File.unique().shape)

        func = lambda x, theta: model(x, theta, a1, a2, a3, a4)

        popt, pcov  = optimization.curve_fit(func, 
                                             self.calibrated.sp_freq, 
                                             self.calibrated.cal_v2, 
                                             p0 = p0, 
                                             sigma = self.calibrated.cal_v2sig)

        self.fit = [{'model': func, 'parameters': popt, 'covariance': pcov}]

    def fit_ldmodel_mc(self, config, teff, logg, nsamples=1e5, model=ldd_claret4, p0=np.array([1.0]), wl_sigma = 5e-3, teff_sigma = 100, logg_sigma = 0.3):

        ld_coeffs = get_ldcs(teff,logg,self.calibrated.wl.unique())

        a1 = np.tile(ld_coeffs.a1, self.calibrated.File.unique().shape)
        a2 = np.tile(ld_coeffs.a2, self.calibrated.File.unique().shape)
        a3 = np.tile(ld_coeffs.a3, self.calibrated.File.unique().shape)
        a4 = np.tile(ld_coeffs.a4, self.calibrated.File.unique().shape)

        func = lambda x, theta: model(x, theta, a1, a2, a3, a4)

        popt, pcov  = optimization.curve_fit(func, 
                                             self.calibrated.sp_freq, 
                                             self.calibrated.cal_v2, 
                                             p0 = p0, 
                                             sigma = self.calibrated.cal_v2sig)

        resid = self.calibrated.cal_v2 - func(self.calibrated.sp_freq, *popt)
        chisq = np.sum((resid/self.calibrated.cal_v2sig)**2)
        dof = len(self.calibrated) - len(popt)
        scale = np.sqrt(chisq/dof)
        print(scale)

        result=[]

        for i in np.arange(np.sqrt(nsamples)):
            if i % 10 == 0:
                print(i)

            #Perturb calibrator diameters
            self.caldiams.sample_diameter = self.caldiams.diameter + np.random.normal(size = len(self.caldiams)) * self.caldiams.uncertainty
            #Perturb wavelength
            perturbed_wl = self.df.wl.unique() + np.random.normal() * wl_sigma
            wl_df = pd.DataFrame({'wl': self.df.wl.unique(), 'perturbed_wl': perturbed_wl}).set_index('wl')
            self.df = self.df.assign(sp_freq_sample = self.baselines.loc[self.df.File].bl.values / wl_df.loc[self.df.wl].perturbed_wl.values)
            #Perturb ld coefficients
            ld_coeffs = get_ldcs(teff+np.random.normal()*teff_sigma, logg+np.random.normal()*logg_sigma, self.df.wl.unique())
            
            a1 = np.tile(ld_coeffs.a1, self.calibrated.File.unique().shape)
            a2 = np.tile(ld_coeffs.a2, self.calibrated.File.unique().shape)
            a3 = np.tile(ld_coeffs.a3, self.calibrated.File.unique().shape)
            a4 = np.tile(ld_coeffs.a4, self.calibrated.File.unique().shape)

            func = lambda x, theta: model(x, theta, a1, a2, a3, a4)

            #Re-Calibrate
            self.calibrate(config)
            cal1 = self.calibrated.cal_v2.copy()

            #Fit recalibrated visibiltiies
            popt, pcov  = optimization.curve_fit(func, 
                                                 self.calibrated.sp_freq_sample, 
                                                 self.calibrated.cal_v2, 
                                                 p0 = p0, 
                                                 sigma = self.calibrated.cal_v2sig)

            fit = func(self.calibrated.sp_freq_sample, *popt)

            for j in np.arange(np.sqrt(nsamples)):
                #Perturb observed visibilites according to the covariance matrix multiplied by reduced chi^2
                self.calibrate(config, scale=scale**2)
                noise = self.calibrated.cal_v2 - cal1
                y2 = fit + noise

                #Repeat least squares fit
                popt, pcov  = optimization.curve_fit(func, 
                                                     self.calibrated.sp_freq_sample, 
                                                     y2, 
                                                     p0 = p0, 
                                                     sigma = self.calibrated.cal_v2sig)

                #Store result
                result.append(popt[0])

        #Put things back to normal
        self.caldiams.sample_diameter = self.caldiams.diameter
        self.df.sp_freq_sample = self.df.sp_freq
        ld_coeffs = get_ldcs(teff, logg, self.df.wl.unique())

        a1 = np.tile(ld_coeffs.a1, self.calibrated.File.unique().shape)
        a2 = np.tile(ld_coeffs.a2, self.calibrated.File.unique().shape)
        a3 = np.tile(ld_coeffs.a3, self.calibrated.File.unique().shape)
        a4 = np.tile(ld_coeffs.a4, self.calibrated.File.unique().shape)

        func = lambda x, theta: model(x, theta, a1, a2, a3, a4)

        self.calibrate(config)
        self.calibrated.cal_v2sig = self.calibrated.cal_v2sig*scale

        #Diameter samples
        result = np.asarray(result)

        meandia = np.mean(result)
        sigdia = np.std(result)
        lowdia,mediandia,updia = np.percentile(result,[15.9,50,84.1])


        self.mcfitld = {'model': model, 
                        'ld_coeffs': ld_coeffs, 
                        'samples': result, 
                        'mean': meandia, 
                        'std': sigdia, 
                        'median': mediandia, 
                        'upper': updia-mediandia, 
                        'lower': mediandia-lowdia}


    def fit_mc(self, config, nsamples=1e5, model=ud, p0=np.array([1.0]), fixed=[False], individual=False, wl_sigma = 5e-3):

        if len(fixed) > 1:
            assert (len(fixed) == len(p0)), "Ambiguous which parameters are to be fixed: p0 and fixed must have same length"
            assert (len(fixed) > np.sum(fixed)), "At least one parameter must not be fixed"
        else:
            assert (fixed[0] is False), "At least one parameter must not be fixed"

        if type(model) is str:
            if model == 'ud':
                model = ud
            elif model == 'ld':
                model = ld
            else:
                try:
                    exec('from .models import '+model)
                    model = vars()[model]
                except ImportError as err:
                    print("Function "+model+" is not implemented in the models module")
                    raise

        if np.sum(fixed) > 0:
            parameters = 'x'
            arguments = 'x'
            for idx,p in enumerate(p0):
                if fixed[idx] is True:
                    arguments = arguments + ', '+str(p)
                else:
                    # This will be a variable of the lambda function
                    parameters = parameters + ', a'+str(idx)
                    arguments = arguments + ', a'+str(idx)

            func = eval("lambda "+parameters+": model("+arguments+")",{'model':model})

            p0 = p0[~np.asarray(fixed)]
        else:
            func = model

        if individual is False:
            
            popt, pcov  = optimization.curve_fit(func, 
                                                 self.calibrated.sp_freq, 
                                                 self.calibrated.cal_v2, 
                                                 p0 = p0, 
                                                 sigma = self.calibrated.cal_v2sig)

            resid = self.calibrated.cal_v2 - func(self.calibrated.sp_freq, *popt)
            chisq = np.sum((resid/self.calibrated.cal_v2sig)**2)
            dof = len(self.calibrated) - len(popt)
            scale = np.sqrt(chisq/dof)
            print(scale)

            result=[]

            for i in np.arange(np.sqrt(nsamples)):
                if i % 10 == 0:
                    print(i)

                #Perturb calibrator diameters
                self.caldiams.sample_diameter = self.caldiams.diameter + np.random.normal(size = len(self.caldiams)) * self.caldiams.uncertainty
                #Perturb wavelength
                perturbed_wl = self.df.wl.unique() + np.random.normal() * wl_sigma
                wl_df = pd.DataFrame({'wl': self.df.wl.unique(), 'perturbed_wl': perturbed_wl}).set_index('wl')
                self.df = self.df.assign(sp_freq_sample = self.baselines.loc[self.df.File].bl.values / wl_df.loc[self.df.wl].perturbed_wl.values)
                #Perturb ld coefficients
                #TO-DO

                #Re-Calibrate
                self.calibrate(config)
                cal1 = self.calibrated.cal_v2.copy()

                #Fit recalibrated visibiltiies
                popt, pcov  = optimization.curve_fit(func, 
                                                     self.calibrated.sp_freq_sample, 
                                                     self.calibrated.cal_v2, 
                                                     p0 = p0, 
                                                     sigma = self.calibrated.cal_v2sig)

                fit = func(self.calibrated.sp_freq_sample, *popt)

                for j in np.arange(np.sqrt(nsamples)):
                    #Perturb observed visibilites according to the covariance matrix multiplied by reduced chi^2
                    self.calibrate(config, scale=scale**2)
                    noise = self.calibrated.cal_v2 - cal1
                    y2 = fit + noise

                    #Repeat least squares fit
                    popt, pcov  = optimization.curve_fit(func, 
                                                         self.calibrated.sp_freq_sample, 
                                                         y2, 
                                                         p0 = p0, 
                                                         sigma = self.calibrated.cal_v2sig)

                    #Store result
                    result.append(popt)

            #Put things back to normal
            self.caldiams.sample_diameter = self.caldiams.diameter
            self.df.sp_freq_sample = self.df.sp_freq
            self.calibrate(config)
            self.calibrated.cal_v2sig = self.calibrated.cal_v2sig*scale

            #Diameter samples
            result = np.asarray(result)

            meanres = np.mean(result, axis=0)
            sigres = np.std(result, axis=0)
            lowres,medianres,upres = np.percentile(result,[15.9,50,84.1], axis=0)


            self.mcfit = [{'model': func,
                          'samples': result,
                          'mean': meanres, 
                          'std': sigres, 
                          'median': medianres, 
                          'upper': upres-medianres, 
                          'lower': medianres-lowres}]

        else:

            mcfits = []

            for filename in self.calibrated.File.unique():
                scan = self.calibrated[self.calibrated.File==filename].reset_index()
                popt, pcov  = optimization.curve_fit(func, 
                                                     scan.sp_freq, 
                                                     scan.cal_v2, 
                                                     p0 = p0, 
                                                     sigma = scan.cal_v2sig)

                resid = scan.cal_v2 - func(scan.sp_freq, *popt)
                chisq = np.sum((resid/scan.cal_v2sig)**2)
                dof = len(scan) - len(popt)
                scale = np.sqrt(chisq/dof)

                result=[]

                for i in np.arange(np.sqrt(nsamples)):
                    if i % 10 == 0:
                        print(i)

                    #Perturb calibrator diameters
                    self.caldiams.sample_diameter = self.caldiams.diameter + np.random.normal(size = len(self.caldiams)) * self.caldiams.uncertainty
                    #Perturb wavelength
                    perturbed_wl = self.df.wl.unique() + np.random.normal() * wl_sigma
                    wl_df = pd.DataFrame({'wl': self.df.wl.unique(), 'perturbed_wl': perturbed_wl}).set_index('wl')
                    self.df = self.df.assign(sp_freq_sample = self.baselines.loc[self.df.File].bl.values / wl_df.loc[self.df.wl].perturbed_wl.values)
                    #Perturb ld coefficients
                    #TO-DO

                    #Re-Calibrate
                    self.calibrate(config)
                    scan = self.calibrated[self.calibrated.File==filename].reset_index()
                    cal1 = scan.cal_v2.copy()

                    #Fit recalibrated visibiltiies
                    popt, pcov  = optimization.curve_fit(func, 
                                                         scan.sp_freq_sample, 
                                                         scan.cal_v2, 
                                                         p0 = p0, 
                                                         sigma = scan.cal_v2sig)

                    fit = func(scan.sp_freq_sample, *popt)

                    for j in np.arange(np.sqrt(nsamples)):
                        #Perturb observed visibilites according to the covariance matrix multiplied by reduced chi^2
                        self.calibrate(config, scale=scale**2)
                        scan = self.calibrated[self.calibrated.File==filename].reset_index()
                        noise = scan.cal_v2 - cal1
                        y2 = fit + noise

                        #Repeat least squares fit
                        popt, pcov  = optimization.curve_fit(func, 
                                                             scan.sp_freq_sample, 
                                                             y2, 
                                                             p0 = p0, 
                                                             sigma = scan.cal_v2sig)

                        #Store result
                        result.append(popt)

                #Put things back to normal
                self.caldiams.sample_diameter = self.caldiams.diameter
                self.df.sp_freq_sample = self.df.sp_freq
                self.calibrate(config)
                self.calibrated.cal_v2sig = self.calibrated.cal_v2sig*scale

                #Diameter samples
                result = np.asarray(result)

                meanres = np.mean(result, axis=0)
                sigres = np.std(result, axis=0)
                lowres,medianres,upres = np.percentile(result,[15.9,50,84.1], axis=0)

                mcfits.append({'model': func, 
                    'samples': result,
                    'mean': meanres,
                    'std': sigres,
                    'median': medianres,
                    'upper': upres-medianres,
                    'lower': medianres-lowres,
                    'bracket': filename, 
                    'pa': scan.pa[0]})

            self.mcfit = mcfits


    def fit_ellipse(self, p0=np.array([0.6,0.5,1.8])):

        if 'fit' not in self.__dict__:
            raise Exception('Fit individual brackets prior to making xy plot')
        if 'pa' not in self.fit[0]:
            raise Exception('Fit individual brackets prior to making xy plot')

        data = pd.DataFrame(self.fit)
        data = data.assign(radius = data.filter(['parameters']).applymap(lambda x: x[0]/2))
        data = data.assign(sigrad = data.filter(['covariance']).applymap(lambda x: np.sqrt(np.diag(x))[0]/2))
        data_reflection = data.copy()
        data_reflection = data_reflection.assign(pa = data_reflection.pa+180.)
        data = pd.concat([data,data_reflection],ignore_index=True)
        data = data.assign(pa = data.pa*np.pi/180)
        

        popt, pcov  = optimization.curve_fit(ellipse, 
                                             data.pa, 
                                             data.radius, 
                                             p0 = p0, 
                                             sigma = data.sigrad)

        self.ellipsefit = {'model': ellipse, 'parameters': popt, 'covariance': pcov}


    def plot(
        self, 
        ax=None,
        xaxis="sp_freq", 
        column="cal_v2",
        caxis="wl",
        xlabel=None,
        ylabel=None,
        xlim=(0,3e8),
        ylim=(0,1),
        title="",
        fmt='o',
        figsize=(8,6),
        ms=0,
        s=20,
        zorder=3,
        capsize=2,
        cmap = plt.cm.viridis,
        **kwargs,
    ) -> matplotlib.axes.Axes: 

        # Default xlabel
        if xlabel is None:
            if xaxis == "sp_freq":
                xlabel = r"Spatial Frequency (rad$^{-1}$)"
            else:
                xlabel = r"Wavelength ($\mu$m)"

        # Default ylabel
        if ylabel is None:
            ylabel = "Squared Visibility"

        x = self.calibrated[xaxis]
        y = self.calibrated[column]
        try:
            yerr = self.calibrated[f"{column}sig"]
        except KeyError:
            yerr = np.full(len(y),np.nan)

        if ax is None:
            fig, ax = plt.subplots(1,figsize=figsize)

        if np.any(~np.isnan(yerr)):
            ax.errorbar(x=x, y=y, yerr=yerr, ms=ms, zorder=zorder, capsize=capsize, fmt=fmt, **kwargs)
            if caxis == 'wl':
                ax.scatter(x,y,c=self.calibrated.wl*1e3, edgecolors='k', cmap=cmap, s=s, zorder=zorder+1)
            else:
                ax.scatter(x,y,c=self.calibrated[caxis], edgecolors='k', cmap=cmap, s=s, zorder=zorder+1)
        else:
            ax.plot(x,y, **kwargs)

        if 'mcfitld' in self.__dict__:
            x = np.linspace(1e-8,xlim[1],num=100)
            ys = []
            for row in self.mcfitld['ld_coeffs'].iterrows():
                ys.append(self.mcfitld['model'](x, self.mcfitld['median'], row[1].a1, row[1].a2, row[1].a3, row[1].a4))
            lines = np.zeros((len(self.mcfitld['ld_coeffs']),100,2))
            lines[:,:,1] = np.asarray(ys)
            lines[:,:,0] = x
            z = 1e3*self.mcfitld['ld_coeffs'].index
            lines = LineCollection(lines, array=z, cmap=cmap)
            ax.add_collection(lines)
            fig.colorbar(lines, label='Wavelength (nm)')
            ax.autoscale()

        elif 'mcfit' in self.__dict__:
            x = np.linspace(1e-8,xlim[1],num=100)
            if 'pa' in self.mcfit[0]:
                ys = []
                pas = []
                for fit in self.mcfit:
                    ys.append(fit['model'](x, *fit['median']))
                    pas.append(fit['pa'])
                lines = np.zeros((len(self.mcfit),100,2))
                lines[:,:,1] = np.asarray(ys)
                lines[:,:,0] = x
                z = pas
                lines = LineCollection(lines, array=z, cmap=cmap)
                ax.add_collection(lines)
                fig.colorbar(lines, label='Position angle (deg)')
                ax.autoscale()

        elif 'fit' in self.__dict__:
            x = np.linspace(1e-8,xlim[1],num=100)
            for fit in self.fit:
                plt.plot(x, fit['model'](x, *fit['parameters']))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        return ax

    def plotxy(
        self, 
        ax=None,
        xlabel='x (mas)',
        ylabel='y (mas)',
        xlim=(-0.6,0.6),
        ylim=(-0.6,0.6),
        title="",
        fmt='C1o',
        **kwargs,
    ) -> matplotlib.axes.Axes:

        if 'fit' not in self.__dict__:
            raise Exception('Fit individual brackets prior to making xy plot')
        if 'pa' not in self.fit[0]:
            raise Exception('Fit individual brackets prior to making xy plot')

        if ax is None:
            fig, ax = plt.subplots(1)

        for fit in self.fit:
            x = 0.5*fit['parameters'][0]*np.cos(fit['pa']*np.pi/180.)
            y = 0.5*fit['parameters'][0]*np.sin(fit['pa']*np.pi/180.)
            ax.plot(x,y,fmt)
            ax.plot(-x,-y,fmt)

        if 'ellipsefit' in self.__dict__:
            x = np.linspace(0,2*np.pi,num=360)
            y = self.ellipsefit['model'](x, *self.ellipsefit['parameters'])
            plt.plot(y*np.cos(x),y*np.sin(x))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_aspect('equal','box')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        return ax
