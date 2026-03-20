#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is a python rendition of Jon Hillier's synthetic spectra IDL script

__author__      = Ethan Ayari, Institute for Modeling Plasmas, Atmospheres and Cosmic Dust
"""


import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.ticker import LogLocator, NullFormatter
from collections import defaultdict

# Improve figure defaults for publication output
plt.rcParams.update({
    "figure.figsize": (7.0, 4.4),
    "figure.dpi": 220,
    "savefig.dpi": 600,
    "figure.autolayout": True,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
})


def configure_apj_axes(ax, xmin, xmax):
    """Apply ApJ-like styling to a mass spectrum axis."""
    ax.set_facecolor("white")
    ax.set_xlim(xmin, xmax)
    ax.set_yscale("log")
    ax.set_xlabel(r"Mass (u)")
    ax.set_ylabel(r"Normalized amplitude")

    # Dense major/minor ticks and subtle grid for publication readability.
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=90))
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.grid(which="major", axis="both", linestyle="-", linewidth=0.4, alpha=0.25)
    ax.grid(which="minor", axis="y", linestyle=":", linewidth=0.3, alpha=0.2)


def velocity_probability(symbol, velocity_kms):
    """Deterministic line-appearance weighting from 10/50/90% velocity anchors."""
    elements = {
        "H": (6.6, 8.9, 12.0),
        "C": (6.9, 9.8, 12.7),
        "O": (12.2, 14.4, 17.0),
        "Mg": (4.1, 5.5, 7.4),
        "Al": (4.1, 5.7, 7.8),
        "Si": (6.2, 8.5, 11.6),
        "Ca": (3.9, 4.6, 5.5),
        "Fe": (6.7, 8.4, 10.6),
        "Na": (6.0, 8.0, 10.8),
        "K": (5.5, 7.4, 9.8),
    }
    if symbol not in elements:
        return 1.0

    v10, v50, v90 = elements[symbol]
    vals_x = np.array([v10, v50, v90], dtype=float)
    vals_y = np.array([0.1, 0.5, 0.9], dtype=float)
    coeff = np.polyfit(vals_x, vals_y, 2)
    prob = np.polyval(coeff, velocity_kms)
    return float(np.clip(prob, 0.0, 1.0))


def annotate_isotopes(ax, x_plot, y_plot, isotope_labels):
    """Draw isotope labels while preventing overlaps in the top margin."""
    if not isotope_labels:
        return

    data_axes_transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    top_y = ax.get_ylim()[1]
    base_label_y_axes = 1.02
    tick_fontsize = plt.rcParams.get("xtick.labelsize", 10)
    x_min, x_max = ax.get_xlim()

    sorted_labels = sorted(isotope_labels, key=lambda iso: iso["mass"])
    min_sep = 2.3  # minimum horizontal spacing in u between neighboring labels
    label_x_positions = []
    for idx, iso in enumerate(sorted_labels):
        label_x = float(np.clip(iso["mass"], x_min + 0.4, x_max - 0.4))
        if idx > 0:
            label_x = max(label_x, label_x_positions[-1] + min_sep)
        label_x_positions.append(label_x)

    # Shift the whole set back into axis limits if right edge overflowed.
    overflow = label_x_positions[-1] - (x_max - 0.4)
    if overflow > 0:
        label_x_positions = [max(x_min + 0.4, x - overflow) for x in label_x_positions]

    for idx, iso in enumerate(sorted_labels):
        mass = iso["mass"]
        label = iso["label"]
        label_x = label_x_positions[idx]
        label_y_axes = base_label_y_axes
        idx = int(np.argmin(np.abs(x_plot - mass)))
        peak_y = y_plot[idx]
        ax.vlines(mass, peak_y, top_y, color="#5a5a5a", linewidth=0.5, alpha=0.7)
        ax.plot(
            [mass, label_x],
            [1.0, label_y_axes],
            color="#5a5a5a",
            linewidth=0.5,
            alpha=0.7,
            transform=data_axes_transform,
            clip_on=False,
        )
        ax.text(
            label_x,
            label_y_axes,
            label,
            ha="center",
            va="bottom",
            fontsize=tick_fontsize,
            rotation=90,
            color="#3a3a3a",
            transform=data_axes_transform,
            clip_on=False,
        )

#%%
def safe_div(x,y):
#Function to handle division by zero "Pythonically"
    try:
        return x/y
    except ZeroDivisionError:
        return 0
    
    
#%%
def fetch_abundances():
#Get full list of all elements and their corresponding symbols, masses and abundances
    eleabund = pd.read_csv("../elementabundances.csv",header = 0)
    return eleabund


#%%
def fetch_rsfs():
#Retreve Hillier Relative Sensitivity Factors (Taken from TOF-SIMS)
    rsfs = pd.read_csv("../rel_sens_fac.csv")
    rsfs.columns = ['Name','Sensitivity Factor']
    return rsfs


#%%
def fetch_rocks():
#Retrieve the available elements from the heidelberg experiment and their compositions (up to 8 elements)
    rocks = pd.read_csv("../rocks.csv", header = 0,)
    rocks.columns = ['Mineral','Element1','abundance1','Element2','abundance2','Element3','abundance3','Element4','abundance4','Element5','abundance5','Element6','abundance6','Element7','abundance7','Element8','abundance8']
    return rocks


#%%
def add_noise(signal):

    """
    

    Parameters
    ----------
    signal : Float64 Array
        An initial numerical spectra free from noise

    Returns
    -------
    A synthetic TOF or mass spectra with gaussian "white" noise added throughout. This whitenoise was derived via fourier analysis of Peridot impact ionization spectra on the Hyperdust instrument.

    """

#Based on a specified signal to noise ratio (SNR), select values from a gaussian sample whose mean corresponds to the ratio. 

    # Set a target SNR
    target_snr_db = 5
    peak_sig = np.where(signal[signal>2.7*10e-4])
    # Calculate signal power and convert to dB 
    sig_avg_watts = np.mean(signal)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise_volts = .01*np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(y))
    
    noise_volts[peak_sig] = 0
    # Noise up the original signal
    y_volts = signal - noise_volts
    return y_volts


#%%   
def make_lama(rockarray, percentarray, velocity_kms=None):
   
    """
    

    Parameters
    ----------
    rockarray : String Array
        A list of what minerals are contained in the sample
    percentarray : Float Array
        The relative abundance of each mineral in the sample

    Returns
    -------
    A synthetic TOF or mass spectra

    """
    
    
#Check that each mineral has a specified abundance
    if(len(rockarray)!=len(percentarray)):
       print('ERROR - NUMBER OF ROCKS MUST MATCH PERCENTAGES')
       return None

#Make sure these abundances sum to 100
    if(np.sum(percentarray)!=100):
       print('ERROR - TOTAL PERCENTAGES MUST BE 100')
       return None

    rockarray_s = np.sort(rockarray)
    percentarray_s = np.sort(percentarray)

#Make sure each mineral is only entered once
    if(len(np.unique(rockarray))!=len(rockarray)):
       print('ERROR - PLEASE ONLY USE EACH MINERAL ONCE!')
       return None


    rocks = fetch_rocks()
#Convert mineral abundances to to fractions 
    rocks['abundance1'] =  rocks['abundance1'].astype(float)/100
    rocks['abundance2'] =  rocks['abundance2'].astype(float)/100
    rocks['abundance3'] =  rocks['abundance3'].astype(float)/100
    rocks['abundance4'] =  rocks['abundance4'].astype(float)/100
    rocks['abundance5'] =  rocks['abundance5'].astype(float)/100
    rocks['abundance6'] =  rocks['abundance6'].astype(float)/100
    rocks['abundance7'] =  rocks['abundance7'].astype(float)/100
    rocks['abundance8'] =  rocks['abundance8'].astype(float)/100
    percentarray_s = percentarray_s/100.0
    
    elems_present = []
    pres_abunds = []

#Unwrap mineral data
    for i in range(len(rockarray)):
#Find what mineral(s) are present
       min_pres = rocks.loc[rocks['Mineral']==rockarray_s[i]]  
#Go throguh and add the elements and their respective abundances for each mineral detected in the sample
       if not (min_pres['Element1'].empty):
          elems_present.append(str(min_pres['Element1']))
       if not (min_pres['abundance1'].empty):
          pres_abunds.append(float((min_pres['abundance1'] + (percentarray_s[i]*min_pres['abundance1'])).iloc[0]))
       if not (min_pres['Element2'].empty):
          elems_present.append(str(min_pres['Element2']))
       if not (min_pres['abundance2'].empty):
          pres_abunds.append(float((min_pres['abundance2'] + (percentarray_s[i]*min_pres['abundance2'])).iloc[0]))
       if not (min_pres['Element3'].empty):
          elems_present.append(str(min_pres['Element3']))
       if not (min_pres['abundance3'].empty):
          pres_abunds.append(float((min_pres['abundance3'] + (percentarray_s[i]*min_pres['abundance3'])).iloc[0]))
       if not (min_pres['Element4'].empty):
          elems_present.append(str(min_pres['Element4']))
       if not (min_pres['abundance4'].empty):
          pres_abunds.append(float((min_pres['abundance4'] + (percentarray_s[i]*min_pres['abundance4'])).iloc[0]))
       if not (min_pres['Element5'].empty):
          elems_present.append(str(min_pres['Element5']))
       if not (min_pres['abundance5'].empty):
          pres_abunds.append(float((min_pres['abundance5'] + (percentarray_s[i]*min_pres['abundance5'])).iloc[0]))
       if not (min_pres['Element6'].empty):
          elems_present.append(str(min_pres['Element6']))
       if not (min_pres['abundance6'].empty):
          pres_abunds.append(float((min_pres['abundance6'] + (percentarray_s[i]*min_pres['abundance6'])).iloc[0]))
       if not (min_pres['Element7'].empty):
          elems_present.append(str(min_pres['Element7']))
       if not (min_pres['abundance7'].empty):
          pres_abunds.append(float((min_pres['abundance7'] + (percentarray_s[i]*min_pres['abundance7'])).iloc[0]))
       if not (min_pres['Element8'].empty):
          elems_present.append(str(min_pres['Element8']))
       if not (min_pres['abundance8'].empty):
          pres_abunds.append(float((min_pres['abundance8'] + (percentarray_s[i]*min_pres['abundance8'])).iloc[0]))

    #print("Elements present:", elems_present)
    #print("Present abundances:", pres_abunds)
    
    newels = []
    #Remove nonsense characters from element names and add them to a list
    for i in range(len(elems_present)):
        tmp1 = elems_present[i].replace('\nName: Element'+str(i+1)+', dtype: object', '')
        tmp2 = re.sub(r'^.*?    ', "", tmp1)
        newels.append(tmp2)
    
    #print("newels: ", newels)
    iso_abun = []
    iso_mass = []
    iso_syms = []
    isotope_data = fetch_abundances()


    for i in range(len(newels)):
        
       #Find what isoptopes are present in the elements and their relative abundances
       iso_pres = isotope_data.loc[isotope_data["Symbol"] == newels[i]]

       #Check for all 8 possible abbundances and add them to a list, along with their corresponding masses
       if not (iso_pres['Abundance1(%)'].empty):
           iso_syms.append(iso_pres['Symbol'])
           iso_abun.append((iso_pres['Abundance1(%)'].astype(float)/100.0)*pres_abunds[i])
       if not (iso_pres['Mass1(u)'].empty):
           iso_mass.append(iso_pres['Mass1(u)'].astype(float))
       
       if not (iso_pres['Abundance2(%)'].empty):
           iso_syms.append(iso_pres['Symbol'])
           iso_abun.append((iso_pres['Abundance2(%)'].astype(float)/100.0)*pres_abunds[i])
       if not (iso_pres['Mass2(u)'].empty):
           iso_mass.append(iso_pres['Mass2(u)'].astype(float))
       
       if not (iso_pres['Abundance3(%)'].empty):
           iso_syms.append(iso_pres['Symbol'])
           iso_abun.append((iso_pres['Abundance3(%)'].astype(float)/100.0)*pres_abunds[i])
       if not (iso_pres['Mass3(u)'].empty):
           iso_mass.append(iso_pres['Mass3(u)'].astype(float))
       
       if not (iso_pres['Abundance4(%)'].empty):
           iso_syms.append(iso_pres['Symbol'])
           iso_abun.append((iso_pres['Abundance4(%)'].astype(float)/100.0)*pres_abunds[i])
       if not (iso_pres['Mass4(u)'].empty):
           iso_mass.append(iso_pres['Mass4(u)'].astype(float))
       
       if not (iso_pres['Abundance5(%)'].empty):
           iso_syms.append(iso_pres['Symbol'])
           iso_abun.append((iso_pres['Abundance5(%)'].astype(float)/100.0)*pres_abunds[i])
       if not (iso_pres['Mass5(u)'].empty):
           iso_mass.append(iso_pres['Mass5(u)'].astype(float))
       
       if not (iso_pres['Abundance6(%)'].empty):
           iso_syms.append(iso_pres['Symbol'])
           iso_abun.append((iso_pres['Abundance6(%)'].astype(float)/100.0)*pres_abunds[i])
       if not (iso_pres['Mass6(u)'].empty):
           iso_mass.append(iso_pres['Mass6(u)'].astype(float))
       
       if not (iso_pres['Abundance7(%)'].empty):
           iso_syms.append(iso_pres['Symbol'])
           iso_abun.append((iso_pres['Abundance7(%)'].astype(float)/100.0)*pres_abunds[i])
       if not (iso_pres['Mass7(u)'].empty):
           iso_mass.append(iso_pres['Mass7(u)'].astype(float))
      
       if not (iso_pres['Abundance8(%)'].empty):
           iso_syms.append(iso_pres['Symbol'])
           iso_abun.append((iso_pres['Abundance8(%)'].astype(float)/100.0)*pres_abunds[i])
       if not (iso_pres['Mass8(u)'].empty):
           iso_mass.append(iso_pres['Mass8(u)'].astype(float))
       
       if not (iso_pres['Abundance9(%)'].empty):
           iso_syms.append(iso_pres['Symbol'])
           iso_abun.append((iso_pres['Abundance9(%)'].astype(float)/100.0)*pres_abunds[i])
       if not (iso_pres['Mass9(u)'].empty):
           iso_mass.append(iso_pres['Mass9(u)'].astype(float))
       
       if not (iso_pres['Abundance10(%)'].empty):
           iso_syms.append(iso_pres['Symbol'])
           iso_abun.append((iso_pres['Abundance10(%)'].astype(float)/100.0)*pres_abunds[i])
       if not (iso_pres['Mass10(u)'].empty):
           iso_mass.append(iso_pres['Mass10(u)'].astype(float))

#Clean isotopic mass values
    for lp in range(len(iso_mass)):
        if(len(iso_mass[lp] != 0)):
           newsym = re.sub(r'^.*?    ', "", str(iso_mass[lp]))
           #print(newsym)
           twosym = newsym.split("\n",1)[0]
           newsym = re.sub(r'^.*?   ', "", twosym)
           if not(str(newsym)=="NaN"):
               iso_mass[lp] = float(newsym)
           else:
               iso_mass[lp] = 0 
#Clean isotopic abundance values              
    for lp in range(len(iso_abun)):
        if(len(iso_abun[lp] != 0)):
           newsym = re.sub(r'^.*?    ', "", str(iso_abun[lp]))
           #print(newsym)
           twosym = newsym.split("\n",1)[0]
           newsym = re.sub(r'^.*?   ', "", twosym)
           if not(str(newsym)=="NaN"):
               iso_abun[lp] = float(newsym)
           else:
               iso_abun[lp] = 0           
    
#Scale isotopic abundances by 100  
    new_abun = []
    for item in iso_abun:
        new_abun.append(item*100.0)
        
#Molar concentration is given by dividing the isotopic abundances by the masses
    molar_conc = [safe_div(i,j) for i, j in zip(new_abun, iso_mass)]
    totalconc = 0.0
    
#Calculate total concentration to normalize
    for i in range(len(molar_conc)):
        if not (molar_conc[i]==0):
            totalconc += float(molar_conc[i])

    molar_conc_norm = []
    
    for item in molar_conc:
        molar_conc_norm.append(item/totalconc)    

#Time to add the Relative Sensitivity Factors
    rsfs = fetch_rsfs()
    
#extract Oxygen sensitivity factor for later normalization
    oxrow = rsfs[rsfs['Name'] == "O"]
    oxsens = oxrow["Sensitivity Factor"]
    oxsens = oxsens.to_numpy()[0]

    
    sens_names= []
    
#Clean Relative Sensitivity Factor names
    for lp in rsfs["Name"]:
        newsym = re.sub(r'^.*?     ', "", str(lp))
        sens_names.append(newsym)
        #print(len(newsym))
    rsfs['Name'] = [sens_names[i] for i in rsfs.index]
    

#Iterate through all pesent isotopes and scale their normalized molar concentrates by their corresponding sensitivity factor
    new_isosyms = []
    for lp in iso_syms:
       newsym = re.sub(r'^.*?    ', "", str(lp))
       twosym = newsym.split("\n",1)[0]
       new_isosyms.append(twosym)

    rsf_pres = []
    rsf_names = []
    for lp in range(len(new_isosyms)):
       pres_rsf = rsfs.loc[rsfs['Name'] == new_isosyms[lp]]

       rsf_pres.append(pres_rsf["Sensitivity Factor"].astype(float))
       rsf_names.append(pres_rsf["Name"].astype(str))



    
#Clean rsf values
    n_rsf_vals = []
    for lp in range(len(rsf_pres)):
       newsym = re.sub(r'^.*?    ', "", str(rsf_pres[lp]))
       twosym = newsym.split("\n",1)[0]

       if("Series" in twosym):
           n_rsf_vals.append(0)
       else:
           n_rsf_vals.append(float(twosym)/oxsens)
    
#clean rsf names, annoying "Series" string persists during Pandas to Numpy conversion
    n_rsf_names = []
    for lp in range(len(rsf_names)):
       newsym = re.sub(r'^.*?    ', "", str(rsf_names[lp]))
       twosym = newsym.split("\n",1)[0]

       if("Series" in twosym):
           n_rsf_names.append("")
       else:
           n_rsf_names.append(twosym)

    
    n_isosyms = []
    for lp in range(len(iso_syms)):
       newsym = re.sub(r'^.*?    ', "", str(iso_syms[lp]))
       twosym = newsym.split("\n",1)[0]
       #print(len(twosym))
       if("Series" in twosym):
           n_isosyms.append("")
       else:
           n_isosyms.append(twosym)
   
#Find the hydrogen index (if it exists) to later disregard
    try:
        hydrodex = n_isosyms.index('H')
    except ValueError:
        hydrodex = -1
   
    for lp in range(len(n_rsf_names)):
        molar_conc_norm[lp] = molar_conc_norm[lp]*float(n_rsf_vals[lp])
        if velocity_kms is not None:
            molar_conc_norm[lp] *= velocity_probability(n_isosyms[lp], velocity_kms)
 

#calculate silver (Ag) target reference peaks
    lama_abund = np.array(molar_conc_norm).flatten().transpose()
    max_abund = max(lama_abund)
    ag_index = isotope_data.loc[isotope_data['Name']=='Silver']
    ag107_amp = float(((max_abund*ag_index['Abundance1(%)'])/100.0).iloc[0])
    ag109_amp = float(((max_abund*ag_index['Abundance2(%)'])/100.0).iloc[0])
    
#append ag refernece values to isotope mass and abundance arrays
    iso_mass.append(107.0)
    iso_mass.append(109.0)
    n_isosyms.append("Ag")
    n_isosyms.append("Ag")
 
    lama_abund = np.append(lama_abund,ag107_amp)
    lama_abund = np.append(lama_abund,ag109_amp)
#lama_abund is now an array of arrival times

#Use the stretch and shift parameters of the instrument to convert isotopic abundance array to a TOF then mass spectra
    stretch = 1800.00 #units of ns per sqrt(mass)
    shift = 0.0 #What does this correspond to physically?
    srate = 2.0 #Sampling rate in ns
    
#Create TOF
    spectrum_t = np.zeros(10000)
    iso_mass = np.array(iso_mass).flatten().transpose() #MUST be a 1-D Numpy array

    peak_times = []
    for lp in range(len(iso_mass)):
        peak_times.append((stretch*np.sqrt(iso_mass[lp]+shift)).astype(float)) #in nanoseconds
    peak_times = np.array(peak_times)
    
    
#Find indexes of time peaks and normalize them by the sampling rate
    peak_positions = []
    for lp in range(len(peak_times)):
       peak_positions.append(np.floor(peak_times[lp]/srate))
    peak_positions = np.array(peak_positions)

#Put arrival times into the peak positions    
    for lp in range(len(lama_abund)):
          spectrum_t[peak_positions[lp].astype(int)] = lama_abund[lp]
          
#Regular Gaussian sample (taken from IDL's gaussian_function)
    gx = np.array([0.0081887, 0.011109 , 0.0149208, 0.0198411, 0.0261214, 0.0340475,
       0.0439369, 0.0561348, 0.0710054, 0.0889216, 0.110251 , 0.135335 ,
       0.164474 , 0.197899 , 0.235746 , 0.278037 , 0.324652 , 0.375311 ,
       0.429557 , 0.486752 , 0.546074 , 0.606531 , 0.666977 , 0.726149 ,
       0.782705 , 0.83527  , 0.882497 , 0.923116 , 0.955997 , 0.980199 ,
       0.995012 , 1.       , 0.995012 , 0.980199 , 0.955997 , 0.923116 ,
       0.882497 , 0.83527  , 0.782705 , 0.726149 , 0.666977 , 0.606531 ,
       0.546074 , 0.486752 , 0.429557 , 0.375311 , 0.324652 , 0.278037 ,
       0.235746 , 0.197899 , 0.164474 , 0.135335 , 0.110251 , 0.0889216,
       0.0710054, 0.0561348, 0.0439369, 0.0340475, 0.0261214, 0.0198411,
       0.0149208, 0.011109 , 0.0081887])

#Convove each peak with the Gaussian sample for more realistic shapes
    real_spectrum_t = np.convolve(spectrum_t,gx) + 2.0
    domain = (((np.arange(10000)*2)-shift)/stretch)**2.0
    spec_max = max(real_spectrum_t)
    real_spectrum_t = real_spectrum_t/spec_max

    isotope_report = []
    for sym, mass, amp in zip(n_isosyms, iso_mass, lama_abund):
        if sym and amp > 0:
            isotope_report.append({"label": f"{int(round(mass))}{sym}", "mass": float(mass), "amplitude": float(amp)})

    return domain, real_spectrum_t, isotope_report


def mineral_formula_from_rocks(rock_row, isotope_data):
    """Estimate empirical formula from weight percentages in rocks.csv."""
    parts = []
    mole_values = []

    for i in range(1, 9):
        elem = rock_row.get(f"Element{i}")
        abund = rock_row.get(f"abundance{i}")
        if pd.isna(elem) or pd.isna(abund):
            continue
        elem = str(elem).strip()
        abund = float(abund)
        iso_pres = isotope_data.loc[isotope_data["Symbol"] == elem]
        if iso_pres.empty:
            continue
        atomic_mass = float(iso_pres["Mass1(u)"].iloc[0])
        moles = abund / atomic_mass
        mole_values.append((elem, moles))

    if not mole_values:
        return "Unknown"

    min_moles = min(m for _, m in mole_values if m > 0)
    for elem, moles in mole_values:
        ratio = moles / min_moles
        if abs(ratio - round(ratio)) < 0.05:
            txt = str(int(round(ratio)))
        else:
            txt = f"{ratio:.2f}"
        parts.append(f"{elem}{txt if txt != '1' else ''}")
    return "".join(parts)


def print_isotope_summary(mineral, formula, isotope_report, velocity_kms=None):
    context = f" @ {velocity_kms} km/s" if velocity_kms is not None else ""
    print(f"\nMineral: {mineral}{context}")
    print(f"Formula: {formula}")
    print("Constituent isotopes (relative abundances):")
    total_amp = sum(i["amplitude"] for i in isotope_report) or 1.0
    by_iso = defaultdict(float)
    for iso in isotope_report:
        by_iso[iso["label"]] += iso["amplitude"]
    for label, amp in sorted(by_iso.items(), key=lambda item: (-item[1], item[0])):
        print(f"  - {label}: {100.0 * amp / total_amp:.2f}%")




if __name__ == "__main__":
    rocks = fetch_rocks()
    isotope_data = fetch_abundances()
    minerals = rocks["Mineral"].dropna().tolist()

    for min_name in minerals:
        rock_row = rocks.loc[rocks["Mineral"] == min_name].iloc[0]
        formula = mineral_formula_from_rocks(rock_row, isotope_data)

        velocity_kms = 20
        x, y, isotopes = make_lama([min_name], [100], velocity_kms=velocity_kms)
        y = y[:-62]
        x_plot = x[:len(y)]
        y_plot = np.clip(y, 1e-6, None)

        print_isotope_summary(min_name, formula, isotopes, velocity_kms=velocity_kms)

        fig, ax = plt.subplots()
        configure_apj_axes(ax, 0, float(np.nanmax(x_plot)))
        ax.plot(x_plot, y_plot, lw=1.15, c="#1f77b4")
        annotate_isotopes(ax, x_plot, y_plot, isotopes)
        ax.text(0.02, 0.96, f"{velocity_kms} km/s", transform=ax.transAxes, ha="left", va="top", fontsize=9)

        output_png = f"../figures/single_mineral_spectra/{min_name}_apj.png"
        output_pdf = f"../figures/single_mineral_spectra/{min_name}_apj.pdf"
        fig.savefig(output_png, bbox_inches="tight")
        fig.savefig(output_pdf, bbox_inches="tight")
        plt.close(fig)

    comparison_minerals = [minerals[0], minerals[1]]
    velocities = [5, 10, 15, 20]

    for mineral in comparison_minerals:
        rock_row = rocks.loc[rocks["Mineral"] == mineral].iloc[0]
        formula = mineral_formula_from_rocks(rock_row, isotope_data)
        fig, axs = plt.subplots(4, 1, figsize=(8.5, 10.5), sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0.55)

        for idx, (ax, vel) in enumerate(zip(axs, velocities)):
            x, y, isotopes = make_lama([mineral], [100], velocity_kms=vel)
            y = y[:-62]
            x_plot = x[:len(y)]
            y_plot = np.clip(y, 1e-6, None)
            print_isotope_summary(mineral, formula, isotopes, velocity_kms=vel)
            configure_apj_axes(ax, 0, float(np.nanmax(x_plot)))
            ax.plot(x_plot, y_plot, lw=1.0, c="#1f77b4")
            annotate_isotopes(ax, x_plot, y_plot, isotopes)
            ax.text(0.02, 0.96, f"{vel} km/s", transform=ax.transAxes, ha="left", va="top", fontsize=9)
            if idx < len(velocities) - 1:
                ax.set_xlabel("")

        output_png = f"../figures/single_mineral_spectra/{mineral}_velocity_grid.png"
        output_pdf = f"../figures/single_mineral_spectra/{mineral}_velocity_grid.pdf"
        fig.savefig(output_png, bbox_inches="tight")
        fig.savefig(output_pdf, bbox_inches="tight")
        plt.close(fig)
