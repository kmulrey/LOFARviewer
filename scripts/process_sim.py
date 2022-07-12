import numpy as np
from optparse import OptionParser
import pickle
import re
from scipy.signal import hilbert
from scipy.signal import resample
import scipy.fftpack as fftp
import os

import process_func as prf
#import antenna_model
import fluence as flu


import sys
#sys.path.insert(1,'/Users/kmulrey/antenna_model/LOFAR_antenna_model/')
sys.path.insert(1,'/vol/astro7/lofar/kmulrey/antenna_model/')
import antenna_model

def measurePulseEnergy(timeseries, half_width, peak_time):
    # Input: timeseries, as 2D numpy-array of shape (npol, nsamples) where npol = # polarizations
    # Pulse power is integrated from the peak position, including 'half_width' samples to each side
    # peak_time = the time of the maximum in the strongest polarization
    
    # rotate the time series to have the peak position at sample 'half_width'
    shifted_timeseries = np.roll(timeseries, int(half_width-peak_time), axis=1)
    # so we can integrate from sample 0 up to (incl) 2*half_width, i.e. 2*half_width+1 samples
    power = np.sum(np.square(shifted_timeseries[:, 0:(2*half_width+1)]), axis=-1) * 5.0e-9
    # Power is from voltage, i.e. after applying antenna model. Therefore, no Z0 factor

    return power
    
    
    
def integrate_power(voltage,lowco=30, hico=80,tstep=0.1e-9):
    # made the power integration a separate function

    dlength=int(len(voltage[0]))
    instr_spec=np.fft.rfft(voltage.T,axis=0)
    frequencies = np.fft.rfftfreq(dlength, tstep) # Ends at 5000 MHz as it should for tstep=0.1 ns

    freqstep = (frequencies[1] - frequencies[0]) / 1.0e6 # MHz
    
    fb = int(np.floor(lowco/freqstep))
    lb = int(np.floor(hico/freqstep)+1)
    window = np.zeros([1,int(dlength/2+1)])
    window[0,fb:lb+1]=1
    
    
    pow0=np.abs(instr_spec[:,0])*np.abs(instr_spec[:,0])
    pow1=np.abs(instr_spec[:,1])*np.abs(instr_spec[:,1])
    
    
    
    maxfreqbin= int(np.floor(tstep/5e-9 * dlength/2.)+1) # Apply frequency bandpass only up to 100 MHz i.e. LOFAR maximum
    #shortspec=np.array([instr_spec[0:maxfreqbin,0]*window[0,0:maxfreqbin,0],instr_spec[0:maxfreqbin,1]*window[0,0:maxfreqbin,0]])
    shortspec=np.array([instr_spec[0:maxfreqbin,0]*window[0,0:maxfreqbin],instr_spec[0:maxfreqbin,1]*window[0,0:maxfreqbin]])
    
    filt=np.fft.irfft(shortspec, axis=-1)
    dlength_new=filt.shape[1]
    
    
    filt *= 1.0*dlength_new/dlength
    # to calculate the time of arrival upsample with a factor 5
    # zelfde factor als in pipeline!!!!
    
     # after downsampling, renormalize the signal!
        
    filt_upsampled=resample(filt,16*dlength_new,axis=-1)
    # compute hilbert enevelope
    hilbenv=np.abs(hilbert(filt,axis=-1))
    hilbenv_upsampled=np.abs(hilbert(filt_upsampled,axis=-1))
    # peak_time is the bin where the maximum is located; NOT the actual time of the peak!
    peak_bin=np.argmax(hilbenv,axis=-1)
    peak_time=np.argmax(hilbenv_upsampled,axis=-1)*1e-9 #in seconds
    peak_amplitude=np.max(hilbenv_upsampled,axis=-1)
    if (peak_amplitude[0]>peak_amplitude[1]):
        pt=peak_bin[0]
    else:
        pt=peak_bin[1]
    # for 3 different window size, the total power is calculated. The window is allowed to `wrap around', so some voodoo is needed to determine the range:
        
    d=filt.shape[1]
    rng=5
    a=int(np.round(np.max([0,pt-rng])))
    b=int(np.round(pt+rng+1))
    c=int(np.round(np.min([d,pt+d-rng])))
    #power11[j]=(np.sum(np.square(filt[:,a:b]),axis=-1)+np.sum(np.square(filt[:,c:d]),axis=-1))*5e-9
  
        
        
    test_power11 = measurePulseEnergy(filt, rng, pt) # This is the one to use (AC)
    #power11[j] = test_power11 # so put it into power11 array
    return test_power11
        

def ProcessData(datadir,fileno,lowco,hico):
    #lowco=30
    #hico=80
    nantennas=160
    l_trace=4082

    integrated_power=np.zeros([nantennas,2])
    antenna_position=np.zeros([nantennas,3])
    xyz_trace=np.zeros([nantennas,3,l_trace])
    trace_01=np.zeros([nantennas,2,l_trace])
    onsky_trace=np.zeros([nantennas,2,l_trace])

    voltage=np.zeros([nantennas,2,l_trace])

    filteredpower=np.zeros([nantennas,2])
    power=np.zeros([nantennas,2])
    power11=np.zeros([nantennas,2])
    power21=np.zeros([nantennas,2])
    power41=np.zeros([nantennas,2])
    peak_time=np.zeros([nantennas,2])
    peak_bin=np.zeros([nantennas,2])
    peak_amplitude=np.zeros([nantennas,2])
    pol_angle=np.zeros([nantennas])
    pol_angle_filt=np.zeros([nantennas])
    fluence=np.zeros([nantennas])
    e_time=np.zeros([l_trace])


    longfile = '{0}/DAT{1}.long'.format(datadir,str(fileno).zfill(6))
    steerfile = '{0}/steering/RUN{1}.inp'.format(datadir,str(fileno).zfill(6))
    listfile = open('{0}/steering/SIM{1}.list'.format(datadir,str(fileno).zfill(6)))
    lorafile = '{0}/DAT{1}.lora'.format(datadir,str(fileno).zfill(6))
    longdata=np.genfromtxt(longfile, skip_header=2, skip_footer=5, usecols=(0,2,3))
    xlength=np.argmax(np.isnan(longdata[:,0]))
    
    hillas = np.genfromtxt(re.findall("PARAMETERS.*",open(longfile,'r').read()))[2:]
    zenith=(np.genfromtxt(re.findall("THETAP.*",open(steerfile,'r').read()))[1])*np.pi/180. #rad; CORSIKA coordinates
    azimuth=np.mod(np.genfromtxt(re.findall("PHIP.*",open(steerfile,'r').read()))[1],360)*np.pi/180.  #rad; CORSIKA coordinates
    energy=np.genfromtxt(re.findall("ERANGE.*",open(steerfile,'r').read()))[1] #GeV
   

    lines = listfile.readlines()
      
    
    for j in np.arange(nantennas):
    #for j in np.arange(10,11):

        antenna_position[j] = (lines[j].split(" ")[2:5]) #read antenna position...
        antenna_file = lines[j].split(" ")[5]   #... and output filename from the antenna list file
        coreasfile = '{0}/SIM{1}_coreas/raw_{2}.dat'.format(datadir,str(fileno).zfill(6),antenna_file[:-1]) #drop the \n from the string!
        data=np.genfromtxt(coreasfile)
        data[:,1:]*=2.99792458e4 # convert Ex, Ey and Ez (not time!) to Volt/meter
        dlength=int(data.shape[0])
        poldata=np.ndarray([dlength,2])
        poldata_filt=np.ndarray([dlength,2])
        dt=data[2,0]-data[1,0]


        az_rot=3*np.pi/2+azimuth    #conversion from CORSIKA coordinates to 0=east, pi/2=north
        zen_rot=zenith
        XYZ=np.zeros([dlength,3])
        XYZ[:,0]=-data[:,2] #conversion from CORSIKA coordinates to 0=east, pi/2=north
        XYZ[:,1]=data[:,1]
        XYZ[:,2]=data[:,3]
        
        # find value to roll
        t = np.argmax([np.max(abs(XYZ[:,0])),np.max(abs(XYZ[:,1])),np.max(abs(XYZ[:,2]))])
        roll_n=int(len(XYZ[:,t])/2-np.argmax(abs(XYZ[:,t])))
        
        # move xyz to center of trace
        xyz_trace[j,0]=np.roll(XYZ[:,0],roll_n)
        xyz_trace[j,1]=np.roll(XYZ[:,1],roll_n)
        xyz_trace[j,2]=np.roll(XYZ[:,2],roll_n)


        poldata[:, 0] = xyz_trace[j,0] * np.cos(zen_rot)*np.cos(az_rot) + xyz_trace[j,1] * np.cos(zen_rot)*np.sin(az_rot) - np.sin(zen_rot)*xyz_trace[j,2]
        poldata[:,1] = np.sin(az_rot)*-1*xyz_trace[j,0] + np.cos(az_rot)*xyz_trace[j,1] # -sin(phi) *x + cos(phi)*y in coREAS 0=positive y, 1=negative x
        
        onsky_trace[j][0]=poldata.T[0]
        onsky_trace[j][1]=poldata.T[1]

        
        pol0fft=np.fft.rfft(poldata.T[0])
        pol1fft=np.fft.rfft(poldata.T[1])
        freq = np.fft.rfftfreq(len(poldata), d=data[1,0]-data[0,0])
        # if you want to change frequency band, change the frequencies here
        bandpass_filter = prf.simple_bandpass(freq,lower_freq=lowco*1e6, upper_freq=hico*1e6)
        pol0fft_filt=pol0fft*bandpass_filter
        
        pol1fft_filt=pol1fft*bandpass_filter
        poldata_filt.T[0]=np.fft.irfft(pol0fft_filt)
        poldata_filt.T[1]=np.fft.irfft(pol1fft_filt)
        
        trace_01[j][0]= poldata_filt.T[0]
        trace_01[j][1]= poldata_filt.T[1]

        hold=flu.calculate_energy_fluence_vector(poldata_filt,data.T[0], signal_window=100., remove_noise=True)
        fluence[j]=hold[0]+hold[1]
        
        
        
        # apply antennna model
    direction=np.asarray([zen_rot,az_rot])
    voltage[:,0],voltage[:,1]=antenna_model.apply_model(trace_01[:,0],trace_01[:,1],direction,dt)
        #print(voltage[j,0].shape,trace_01[j][0].shape)
        
    
    for j in np.arange(nantennas):
            integrated_power[j]=integrate_power(voltage[j])
    
     ### convert CORSIKA to AUGER coordinates (AUGER y = CORSIKA x, AUGER x = - CORSIKA y
     
    e_time=np.arange(0,4082,1)*dt

    temp=np.copy(antenna_position)
    antenna_position[:,0], antenna_position[:,1], antenna_position[:,2] = -temp[:,1]/100., temp[:,0]/100., temp[:,2]/100.
    azimuth=3*np.pi/2+azimuth #  +x = east (phi=0), +y = north (phi=90)

    sim_info={'e_time':e_time, 'zenith':zenith, 'azimuth':azimuth, 'energy':energy, 'hillas':hillas, 'antenna_position':antenna_position, 'fluence':fluence, 'xyz_trace':xyz_trace,'poldata':onsky_trace,'poldata_filt':trace_01,'voltage':voltage,'integrated_power':integrated_power}
    
    return sim_info
