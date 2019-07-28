# Successor to trace_training_prepare.py
# Supports processing of an .mseed's stream to extract:
#      n useful channels, class type, and trigger locations for each channel and stores in new stream

import glob
import re
import obspy
import numpy as np
from obspy import UTCDateTime, read, Stream,  Trace

from obspy.signal.trigger import plot_trigger
from obspy.signal.trigger import recursive_sta_lta
from obspy.signal.trigger import carl_sta_trig
from obspy.signal.trigger import classic_sta_lta
from obspy.signal.trigger import delayed_sta_lta
from obspy.signal.trigger import z_detect
from obspy.signal.trigger import trigger_onset
from obspy.signal.filter import remez_fir

from obspy.signal.invsim import simulate_seismometer, corn_freq_2_paz

from obspy.io import mseed
import os
from tqdm import trange
from pca import pca_reduce, plot_wiggle, normalize1D
import pandas as pd
from math import floor, sqrt
import matplotlib.pyplot as plt
from matplotlib.pyplot import legend

from fourier import do_fft, plot_fft, fft2stream, fft2float64
import random
from scipy.signal import find_peaks
class_dict = {'MEQ' : 0, 'CASSM' : 1, 'DRILLING' : 2, 'ERT' : 3, 'NOISE' : 4}
OVERWRITE = 1

unlabeled_mseed_path            = "D:\\trigger\\"
labeled_data_png_path           = "D:\\labeled_data\\png\\"
labeled_data_path               = "D:\\labeled_data\\optimized_and_labeled_triggers\\"
error_log_path                  = "C:\\Users\\David\\Desktop\\training_error_log.txt"

# Parameters to specify resulting optimized stream location
sequentially_load_pngs  = 0
use_agnostic_triggers   = 1
find_fft                = 0

class_wiggles_chopped   = 0
class_wiggles_saved     = 0

plot_wiggles            = 0
verif_amt               = -100
verified                = 0
max_stream_count        = 2444
#useful_channels         = [(10, 'PDB11'), (54, 'OT16')]
useful_channels         = [(10, 'PDB11')] # TODO: optimize general trigger detection for PDB?
useful_channels         += [(54, 'OT16')]

std_on                  = -1
std_off                 = -1

# Returns location of one or more triggers for a supplied .mseed or .dat wiggle.
# For labelling purposes, the class_type is used to apply proper filtering & 
# sta/lta trigger detection if agnostic_approach is not used. if -1, the generalized approach is used.
def return_trigger_index(wiggle, channel_name, class_type):
    global verified
    trace           = wiggle.copy()
    trigger_indices = []
    index_depth = 0.3                                                           # how far into max indices to return possible trigger
                                                                                # (avoids returning triggers at very beginning)
    sta     = 1
    lta     = 1
    on      = 1
    off     = 1
    max_triggers = 25
    max_trigger_length = 20000                                                  # "Maximum length of triggered event in samples.
                                                                                # A new event will be triggered as soon as the signal reaches again above thres1."
    if channel_name == 'PDB11':                                                 # Hydrophone 
        if (class_type == -1):
            trigger_indices, ctf = standard_trigger_finder(trace, channel_name)
            on = std_on
            off = std_off
    elif channel_name == 'OT16':
        if (class_type == -1):
            #trace.filter('bandpass', freqmin=1000, freqmax=15000)
            sta = 20
            lta = 60
            trigger_indices, ctf = standard_trigger_finder(trace, channel_name)
            on = std_on
            off = std_off

    trigger_data = np.zeros(1, dtype=np.int32)
    depth_threshold = 0.03                                                   # triggers at indices sooner than this will not be used
    trigger_index = floor(index_depth * len(trigger_indices))                  # just in case all triggers are very early
    for trigger_pair in trigger_indices:
        if not ( (trigger_pair[0]/len(ctf)) < depth_threshold ):
            trigger_index = trigger_pair[0]
            break

    trigger_data[0] = trigger_index

    if (verified < verif_amt):
        plot_trigger(trace, ctf, on, off)
        verified += 1

        for t, trigger in enumerate(trigger_data):
            #--- testing pca ---
            trigger_offset  = 300
            window_size     = 1300
            block_size      = 2
            pca_n_pts       = floor(window_size/block_size) - 1                     
            trigger_start = trigger# trigger_start_index[0].data[0]                         

            #print("Suspected trigger start: %s depth: %.03f%% in %d" % (str(trigger_start), 100*(trigger_start / ot16_npts), ot16_npts))
            trace_start = (trigger_start - trigger_offset) if ((trigger_start - trigger_offset) > 0) else 0
            trace_end = trace_start + window_size

            last_data_pt = trace[len(trace)-1]
            trace_window    = []
            if (len(trace) < trace_end):
                trace_window = np.pad(trace[trace_start:len(trace)],
                                     (0, (trace_end - len(trace))),
                                     'constant', constant_values=last_data_pt)
            else:
                trace_window = trace[trace_start:trace_end]

            pca, pca_wiggle, reconstructed_wiggle = pca_reduce(trace_window, pca_n_pts, block_size)
            if plot_wiggles:
                plot_wiggle(pca_wiggle)
                continue

            #plot_wiggle(reconstructed_wiggle)

            #--- testing pca ---

    trigger_stream = Stream([Trace(data=trigger_data)])
    return trigger_stream

# deprecated . . . 
def remove_cassm_spike(trace):
    t = trace.copy()
    window_length = 300
    window_start = 0
    window_end = window_start + window_length
    jump_amt = 50
    spike_threshold = 0.04

    for j in range(0, floor(len(t) / window_length)):
        w = t[window_start:window_end]
        nw = normalize1D(w)
        window_smoothness = np.std(nw)
        if (window_smoothness > spike_threshold):
            #t_plot = plt.plot(w, color='#95a172')
            #plt.show()
            return trace[:window_start]

        window_start += jump_amt
        window_end = window_start + window_length
    return trace

# given a trace with a drastic spike towards the end, return truncated trace without this tail
# (used almost exclusively by CASSM wiggles)
def trace_tail_chopper(trace):
    global class_wiggles_chopped
    t = trace.copy()
    max_n_pts = 30
    spine_length = 500                                                          # in cassm, spikes happen after trace values are close to 0 out for some time
    tail_spike_length = 200
    spine_smoothness_threshold = 0.040                                          # points just before tail begins should be very smooth for cassm spike
    tail_spikiness_threshold = 0.02
    tail_start_depth = 0.45                                                     #tails always start towards end of trace
    suspected_tail_start = floor(tail_start_depth * len(t))
    tail_start_buffer = 180                                                      # don't want trace to end RIGHT at the start of tail
    
    extreme_indices = []
    extreme_indices.append(find_index_of_min_val(t[suspected_tail_start:], max_n_pts)[0] + suspected_tail_start)
    extreme_indices.append(find_index_of_max_val(t[suspected_tail_start:], max_n_pts)[0] + suspected_tail_start)
    
    normalized_tail = abs(normalize1D(t[suspected_tail_start:]))
    peaks,_ = find_peaks(normalized_tail, height=None, threshold=None, distance=None, prominence=0.02, width=None)
    
    if (plot_wiggles):
        plt.plot(peaks, normalized_tail[peaks], "ob")
        plt.plot(normalized_tail)
        plt.show()
        #plt.legend(['prominence'])

    # tail spike may only spike upwards, only downwards, up then down, or down then up
    # so instead, check for peak (whether max or min), if spine behind? tail.
    #for p in extreme_indices:
    for p in peaks:
        p += suspected_tail_start
        tail_start = p-tail_start_buffer
        normalized_spine = normalize1D(t[tail_start-spine_length:tail_start])
        normalized_tail_spike = normalize1D(t[tail_start:tail_start+tail_spike_length])
        spine_smoothness = np.std(normalized_spine)
        tail_spikiness = np.std(normalized_tail_spike)
        if (plot_wiggles):
            print("\ntail_start: %d\nspine_std: %0.08f\ntail_std: %.08f" % (p, spine_smoothness, tail_spikiness))
            plt.style.use('dark_background')
            t_plot = plt.plot(normalized_tail_spike, color='#95a172')
            s_splot = plt.plot(normalized_spine, color='#fc8803')
            #plt.legend((t_plot, s_splot), ('tail', 'spine'))
            #plt.show()

        if (spine_smoothness < spine_smoothness_threshold) and (tail_spikiness > tail_spikiness_threshold):
            class_wiggles_chopped += 1
            #print("\ntail chopped! %.01f%% removed" % (100*(1-(tail_start/len(t)))))
            return trace[:tail_start]
    return trace                                                                # else return unaltered trace


def standard_trigger_finder(trace, channel_name):
    global std_on
    global std_off
    max_triggers = 20
    max_trigger_length = 20000
    #ctf_start = 300                                                             # avoids triggering on initial spike
    if (channel_name == 'PDB11'):
        trace.filter('highpass', freq=1500)
        sta = 20
        lta = 60
        ctf = recursive_sta_lta(trace.data, sta, lta)
        #std_on  = ctf[find_index_of_max_val(ctf, max_triggers)] * 0.84
        std_on = ctf[find_index_of_best_val(ctf, max_triggers)] * 0.98

        std_off = std_on * 0.8
        trigger_indices = trigger_onset(ctf, std_on, std_off, max_trigger_length)
    
    if (channel_name == 'OT16'):
        #trace.filter('bandpass', freqmin=1000, freqmax=15000)
        sta = 10
        lta = 50
        ctf = recursive_sta_lta(trace.data, sta, lta)
        #ctf = ctf[ctf_start:]

        #std_on  = ctf[find_index_of_max_val(ctf, max_triggers)] * 0.77
        std_on = ctf[find_index_of_best_val(ctf, max_triggers)] * 0.98
        std_off = std_on * 0.84
        trigger_indices = trigger_onset(ctf, std_on, std_off, max_trigger_length)

        # sta = 20
        # lta = 60
        # ctf = recursive_sta_lta(trace.data, sta, lta)
        # std_on  = ctf[find_index_of_max_val(ctf, max_triggers)] * 0.9
        # std_off = std_on * 0.8

    return trigger_indices, ctf

# Given a characteristic function for the triggers of a trace, return the index
# which corresponds to the peak value
def find_index_of_max_val(ctf, max_triggers):
    max_indices = ctf.argsort()[-max_triggers:][::-1]

    max_indices.sort()
    return np.unravel_index(np.argmax(ctf, axis=None), ctf.shape)

def find_index_of_min_val(wiggle, max_n_pts):
    min_indices = wiggle.argsort()[max_n_pts:][::-1]
    min_indices.sort()

    return np.unravel_index(np.argmin(wiggle, axis=None), wiggle.shape)

# returns index of first value in ctf which is closest to the mean of the top max_triggers
# avoids returning peak value every time
def find_index_of_best_val(ctf, max_triggers):
    ctf_start = 200
    truncated_ctf = ctf[ctf_start:] # removes starting spike that is apparent in nearly all characteristic fxn's
    plt.style.use('dark_background')
    #plt.plot(truncated_ctf, color='#ccd0ff')
    #plt.show()
    max_indices = truncated_ctf.argsort()[-max_triggers:][::-1]
    sum_value = 0
    avg_value = 0
    for i in max_indices:
        sum_value += truncated_ctf[i]
    avg_value = sum_value / max_triggers

    best_index = min(range(len(truncated_ctf)), key=lambda i: abs(truncated_ctf[i]-avg_value)) + ctf_start  # return index of avg_value
    return best_index

# Takes a class type (str) and optimizes all .mseeds based on each .png in class' directory
# New stream includes:
#    a) traces from useful_channels
#    b) detected trigger locations for each channel (agnostic or otherwise)
#    c) class type (using global int32 class_dict mapping)
def optimize_class(class_type):
    global verified
    png_path        = labeled_data_png_path + class_type + "\\*.png"
    
    if sequentially_load_pngs:
        all_class_pngs  = sorted(glob.glob(png_path))
    else:
        all_class_pngs  = glob.glob(png_path)
        random.shuffle(all_class_pngs)

    max_pngs        = max_stream_count if (len(all_class_pngs) > max_stream_count) else len(all_class_pngs)
    class_label     = png_path.split("\\")[png_path.count("\\")-1]              # name of class eg 'ert', 'meq'.. or just use class_type?
    
    missed_pngs      = 0                                                        # counter for missing .mseed files
    handle_errors    = 0                                                        # boolean to deal with 'png not found' errors
    error_msg        = []               
    fft_start        = 50                                                       # hard-coded start index of fft (avoids ubiquitious initial spike)
    fft_end          = 900                                                      # most useful FFT info. occurs earlier than this
    channel_triggers = np.zeros(len(useful_channels), dtype=np.int32)           # List of detected trigger start locations for each useful_channel (in order)
    
    for png in trange(max_pngs, leave=True):
        useful_traces    = []                                                   # List of traces which are representative of entire stream (at least ot16 and pdb11)
        useful_ffts      = []                                                   # FFT for each useful channel?
        try:
            png_name = all_class_pngs[png]
            timestamp = re.search('[0-9]+(\.[0-9][0-9]?)?', png_name).group(0)  # extracts timestamp from file name
        except Exception as e:
            error_msg.append("Could not find PNG [" + str(e) + "]")
            missed_pngs +=1
        try:
            st = read(unlabeled_mseed_path + timestamp + ".mseed")              
            class_label_stream = return_class_stream(class_label, st[0])
        except Exception as e:
            error_msg.append("Could not find corresponding .mseed file for "
                             + class_label + '\\' + timestamp + ".png")
            missed_pngs += 1
            continue

        optimized_mseed_class_path = labeled_data_path
        optimized_mseed_class_path += class_label + "_training\\"
        
        optimized_and_labeled_mseed_path = optimized_mseed_class_path + '\\' + timestamp + '-L.mseed'

        for c, uc in enumerate(useful_channels):                                # create list of traces from representative channels
            t = st[uc[0]].copy()                                                # Load trace from respective useful channel
            useful_traces.append(t)
            channel_name = uc[1]

            #### Important: this trace preprocessing must be replicated when training / classifying in SVM
            trace_start = 300                                                   # skips this many npts to discard initial spike 
            t.filter('lowpass', freq=8000)
            #t.filter('highpass', freq=800)
            t.data = t.data[trace_start:]

            if plot_wiggles and channel_name == 'OT16':
                t.plot(method='full', equal_scale=True, color='#ccd0ff', bgcolor='#07012b', linewidth='1.2', dayPlot=True, number_of_ticks=8, size=(1000,400))#size=(2000,920))
                nt = normalize1D(t.data)
                plt.plot(nt)
                plt.show()
            if channel_name == 'OT16':# and class_label == 'CASSM':               # cassm tails do not exist in PDB11
                #print(t.stats.npts)
                npts_b4 = t.stats.npts
                t.data = trace_tail_chopper(t.data)
                #t.data = remove_cassm_spike(t.data)
                npts_after = t.stats.npts
                #print(t.stats.npts)
                if npts_b4 != npts_after:
                    t.detrend(type='linear')

            if plot_wiggles and channel_name == 'OT16':
                t.plot(method='full', equal_scale=True, color='#26de4e', bgcolor='#07012b', linewidth='1.0', dayPlot=True, number_of_ticks=8, size=(1000,400))#size=(2000,920))

            if find_fft:
                if channel_name == 'OT16':                                      #find fft for OT16
                    filtered_wiggle = t.copy()
                    filtered_wiggle.filter('lowpass', freq=10000)
                    trace_fft = do_fft(filtered_wiggle) 
                    trace_fft = trace_fft[fft_start:fft_end]
                    fft_float64 = fft2float64(trace_fft)
                    #plot_fft(fft_float64)
                    
            if use_agnostic_triggers:
                trigger_stream = return_trigger_index(t, uc[1], -1)             # use agnostic trigger detection (generalized for all classes to better reflect classifying scenario)
            else:
                trigger_stream = return_trigger_index(t, uc[1], class_dict.get(class_type.upper())) # use true class for optimal training

            channel_triggers[c] = trigger_stream[0].data[0]

        if (os.path.isdir(optimized_mseed_class_path) == 0):                    # make directory if nonexistent
            try:
                os.makedirs(optimized_mseed_class_path)
            except OSError:
                error_msg.append("Could not create directory " + optimized_mseed_class_path + " [" + str(e) + "]")
                
        # *********************** Saving optimized stream **********************
        try:
            optimized_stream = Stream()     
            for ut in useful_traces:                                            # order of channels is imperative
                optimized_stream.append(ut)         
                if (verified < verif_amt):
                    #ut.plot(method='full', color='#0798a8', size=(2000,920))
                    #verified += 1
                    pass
            # fft for each channel? 
            if find_fft:
                optimized_stream.append(fft2stream(trace_fft, st[0])[0])            

            optimized_stream.append(list2stream(channel_triggers,st[0])[0])
            optimized_stream.append(class_label_stream[0])
            optimized_stream.write(optimized_and_labeled_mseed_path, format='MSEED')
            class_wiggles_saved += 1
            #found_mseeds += 1
        except Exception as e:
            error_msg.append('Failed to save: ' + optimized_and_labeled_mseed_path + " [" + str(e) +"]")

    if (handle_errors):
        for msg in error_msg:
            print(msg)
        try:
            with open(error_log_path, 'a+') as f:
                for error in error_msg:
                    f.write(str(error) + "\n")
        except Exception as e:
            print("could not open " + error_log_path + " " + str(e))

    #found_mseeds = max_pngs - missed_pngs
    #print("%.04f%% of %s data labeled (%d/%d)" % ((found_mseeds / max_pngs * 100), class_type, found_mseeds, max_pngs))
    #print("{0:{.2}f}%% of {1} data labeled ({2}/{3})".format(100*(found_mseeds / max_pngs), class_type.upper(), found_mseeds, max_pngs))

# Returns a stream that contains semi-relevant header data and the int which maps to the class of an .mseed file
def return_class_stream(class_key, sample_trace):
    label_data = np.zeros(1, dtype=np.int32)
    label_data[0] = class_dict.get(class_key.upper())                           # returns corresponding value from class_dict

    stats = {'network': 'SV',
             'station' : 'ALL',
             'channel' : 'XXX',
             'npts'    : 1,
             'sampling_rate' : '1',
             'starttime' : sample_trace.stats.starttime
    }
    class_label_stream = Stream([Trace(data=label_data, header=stats)])

    return class_label_stream

def list2stream(list2convert, sample_trace):

    stats = {'network': 'SV',
             'station' : 'ALL',
             'channel' : 'XXX',
             'npts'    : len(list2convert),
             'sampling_rate' : '1',
             'starttime' : sample_trace.stats.starttime
    }

    return Stream([Trace(data=list2convert, header=stats)])

def main():
    global verified
    global class_wiggles_chopped 
    global class_wiggles_saved

    classes = ['cassm','drilling','ert','meq']
    #classes = ['drilling']
    for class_type in classes:
        print("Optimizing " + class_type.upper() + " .mseed files.")
        optimize_class(class_type)
        print("chopped %d out of %d traces from %s" % (class_wiggles_chopped, class_wiggles_saved, class_type))
        verified = 0
        class_wiggles_chopped = 0
        class_wiggles_saved = 0

if __name__ == '__main__':
    main()