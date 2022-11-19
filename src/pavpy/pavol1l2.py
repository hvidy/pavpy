import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import scipy.optimize as optimization

from .utils import (
	get_uv,
	get_diams,
	wtmn
)
from .models import (
	ud,
	ld,
	ellipse
)

class PavoObs():
	"""
	Subclass of pandas `~pandas.DataFrame`

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
			                     pa = self.baselines.loc[self.df.File].pa.values)

		self.caldiams = get_diams(self.df)


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
	        exp_v2 = ud(self.df.sp_freq, self.caldiams.loc[star].diameter)
	    
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
	            exp_v2 = ud(self.df.sp_freq, self.caldiams.loc[star].diameter)

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


	def calibrate(self, config):

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
		xlabel=None,
		ylabel=None,
		xlim=(0,3e8),
		ylim=(0,1),
		title="",
		fmt='o',
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
			fig, ax = plt.subplots(1)

		if np.any(~np.isnan(yerr)):
			ax.errorbar(x=x, y=y, yerr=yerr, fmt=fmt, **kwargs)
		else:
			ax.plot(x,y, **kwargs)

		if 'fit' in self.__dict__:
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
