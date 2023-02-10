import pandas as pd
import numpy as np

from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
import astropy.units as units
from astroquery.vizier import Vizier
from dustmaps.bayestar import BayestarQuery, BayestarWebQuery
from scipy.interpolate import griddata

def estimate_theta_vk(vmag,kmag):
    # Calculate angular diameter from Boyajian et al. (2014) relation
    logtheta = 0.26382*(vmag-kmag) + 0.53246 - 0.2 * vmag
    
    return np.power(10,logtheta)

def get_coords(star):
    # Get galactic coordinates for a given star, assuming it is the brightest object in the search
    v = Vizier(columns=["RA_ICRS","DE_ICRS","Plx","Gmag"], catalog="I/355/gaiadr3")
    result = v.query_object(star)
    coords = SkyCoord(result[0][np.argmin(result[0]['Gmag'])]['RA_ICRS']*units.deg, 
                  result[0][np.argmin(result[0]['Gmag'])]['DE_ICRS']*units.deg, 
                  distance=1000/result[0][np.argmin(result[0]['Gmag'])]['Plx']*units.pc,
                 equinox='J2016').transform_to('galactic')
    return coords

def get_extinction(coords,version='bayestar2015'):
    # Get E(B-V) for a given galactic coordinate
    try:
        bayestar = BayestarWebQuery(version=version)
        reddening = bayestar(coords, mode='median')
    except:
        bayestar = BayestarQuery(version=version)
        reddening = bayestar(coords, mode='median')


    return reddening
    
def deredden(vmag,kmag,ebv):
    # Calculate dereddened v and k magnitudes assuming O'Donnell (1994) law
    Rv = 3.1
    av = Rv * ebv
    ak = (0.148 - 0.099/Rv) * av
    v = vmag - av
    k = kmag - ak
    
    return v,k

def get_vkmags(star):
    # Get V and K mags for a given star, and convert Tycho V mag to Johnson V
    v = Vizier(columns=["BTmag","VTmag"], catalog="I/259/tyc2")
    result = v.query_object(star)
    bt_minus_vt = result[0]['BTmag'][0]-result[0]['VTmag'][0]
    
    tycho = pd.read_fwf('src/pavpy/Tycho_BV_Bessel2000.dat',skiprows=2, names=['BtVt','VVt','dBV','VHp'])
    
    vmag = np.interp(bt_minus_vt, tycho.BtVt, tycho.VVt) + result[0]['VTmag'][0]
    
    v = Vizier(columns=["Kmag"], catalog="II/246/out")
    result = v.query_object(star)
    
    kmag = np.min(result[0]['Kmag'])
    
    return vmag,kmag

def get_uv(df):
    #Calculate u, v, projected baseline and position angle for each observation in a dataframe

    #Location of the observatory
    chara = EarthLocation.of_site('CHARA')
    lat = chara.lat.radian #Convert latitutde to radians
        
    #Relative location of telescopes
    tel_locations = pd.read_csv('src/pavpy/tel_locations.txt', delimiter=' ', names=['telname','e_offset','n_offset','z_offset'])
    tel_locations = tel_locations.set_index('telname')
    
    summary = df.groupby(df.File).first()
    
    summary = summary.assign(ra = summary.filter(['Star']).applymap(lambda x : SkyCoord.from_name(x).ra.radian),
                             dec = summary.filter(['Star']).applymap(lambda x : SkyCoord.from_name(x).dec.radian),
                             lst = Time(summary.JD + 2451545, format='jd', location=chara).sidereal_time('mean').radian)
    summary = summary.assign(ha = summary.lst - summary.ra,
                             e_offset = summary.filter(['T1']).applymap(lambda x : tel_locations.loc[x].e_offset).values - 
                                           summary.filter(['T2']).applymap(lambda x : tel_locations.loc[x].e_offset).values,
                             n_offset = summary.filter(['T1']).applymap(lambda x : tel_locations.loc[x].n_offset).values - 
                                           summary.filter(['T2']).applymap(lambda x : tel_locations.loc[x].n_offset).values,
                             z_offset = summary.filter(['T1']).applymap(lambda x : tel_locations.loc[x].z_offset).values - 
                                           summary.filter(['T2']).applymap(lambda x : tel_locations.loc[x].z_offset).values)
    summary = summary.assign(bx = -np.sin(lat)*summary.n_offset + np.cos(lat)*summary.z_offset,
                             by = summary.e_offset,
                             bz = np.cos(lat)*summary.n_offset + np.sin(lat)*summary.z_offset)
    summary = summary.assign(u = np.sin(summary.ha)*summary.bx + np.cos(summary.ha)*summary.by,
                             v = -np.sin(summary.dec)*np.cos(summary.ha)*summary.bx + np.sin(summary.dec)*np.sin(summary.ha)*summary.by 
                                    + np.cos(summary.dec)*summary.bz)
                             # w = np.cos(summary.dec)*np.cos(summary.ha)*summary.bx - np.cos(summary.dec)*np.sin(summary.ha)*by + np.sin(summary.dec)*summary.bz
    summary = summary.assign(bl = np.sqrt(summary.u**2 + summary.v**2),
                             pa = np.arctan2(summary.v,summary.u)*180/np.pi
                            )
    
    # return summary
    return summary.filter(['u','v','bl','pa'])

def get_diams(df, fractional_uncertainty):

    vk_diams = []

    for star in df.Star.unique():
        # Get V and K mags from Tycho-2 and 2MASS. Convert Tycho V to Johnson V
        vmag, kmag = get_vkmags(star)
        # Get Gaia coordinates and parallax
        coords = get_coords(star)
        # Get E(B-V) from Green et al. (2015) dust map
        ebv = get_extinction(coords)
        # De-redden V and K magntiudes
        vmag,kmag = deredden(vmag, kmag, ebv)
        # Calculate LD diamter from V-K relation of Boyajian et al. 2014
        thetaLD = estimate_theta_vk(vmag,kmag)

        vk_diams.append(thetaLD)

    uncertainty = fractional_uncertainty*np.asarray(vk_diams)

    return pd.DataFrame({'star': df.Star.unique(), 'diameter': vk_diams, 'uncertainty': uncertainty, 'sample_diameter': vk_diams}).set_index('star')

def get_ldcs(teff, logg, wavelengths):

    model_coeffs = pd.read_json('stagger_4term_coeffs.json')

    coeffs = []

    for wavelength in wavelengths:
        coeff = griddata(model_coeffs.filter(['teff','logg']).values, 
            np.stack(np.concatenate(model_coeffs.filter([str(wavelength)]).values)), 
            np.array([teff,logg]), 
            method='cubic',
            rescale=True)
        coeffs.append(coeff[0])

    return pd.DataFrame(coeffs, index= wavelengths, columns = ['a1','a2','a3','a4'])


def wtmn(x, sigx):

    return (x*sigx**-2).agg('sum')/(sigx**-2).agg('sum')

def randomcorr(covmat):

    evals, evects = np.linalg.eigh(covmat)
    lin_comb = evects @ np.diag(np.sqrt(evals))
    random_vector = lin_comb @ np.random.normal(size = evals.shape)
    
    return random_vector