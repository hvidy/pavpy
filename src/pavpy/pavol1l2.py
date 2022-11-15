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
	ud
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

	def fit_model(self, model=ud, p0=np.array([1.0])):

		if model is ud:
			fit  = optimization.curve_fit(ud, self.calibrated.sp_freq, self.calibrated.cal_v2, p0, self.calibrated.cal_v2sig)
			return fit
		else:
			return "This should raise an error"


	def plot(
		self, 
		ax=None,
		xaxis="sp_freq", 
		column="cal_v2", 
		xlabel=None,
		ylabel=None,
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

		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)

		ax.set_xlim(0,3e8)
		ax.set_ylim(0,1)

		return ax