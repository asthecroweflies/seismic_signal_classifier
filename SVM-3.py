
import time
import calendar

from math import floor, sqrt
import pandas as pd
import obspy
from obspy import read

import glob
import os
import random
from tqdm import trange
from pca import pca_reduce, plot_wiggle
import numpy as np
import matplotlib.pyplot as plt
import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix

from stream_optimization_labelling_v3 import return_trigger_index, trace_tail_chopper
#from fourier import do_fft, plot_fft, fft2float64
from utc import ts2date, date2ts
import sys
from timeit import default_timer as timer


'''
    SVM Classifier for Microseismic Event Identification

    main:
        - do_train (params: do_svm_grid_search, iter_pca_params)
            - create TrainingFeatures obj. (contains window/block size, and trigger_offset)
            - calls train_with_these_features():
                1) Build Data Set
                    a) Using given TrainingFeatures object, perform PCA from n useful_channels
                    b) conjoin n pca-ified wiggles (ie .extend()), and append to total_data
                    c) extract class_label from optimized/labeled mseed and append to class_labels 

                2) Create Classifier
                    a) Splits total_data into internal testing/validation splits with accompanying y_labels
                    b) Calls sklearn's SVC method to create SVM classifier
                    c) Fits data to classifier to catered data (dimensionality reduction 3D->2D)
                    d) Can grid search through params (mainly C & gamma) to find optimal classifier

                3) Evaluate Classifier
                    a) Finds precision and accuracy scores on both training & validation data sets
                    b) Plots and/or saves confusion matrices created using accuracies on training data set
                    c) Saves precision and accuracy metrics to text-file displaying used PCA params

                4) Save classifier
                    a) With the classfier's name determined by max_stream_count & PCA params, save model
                       using Joblib's serialization library

        - classify_single_mseed
            - calls classify_mseed(stream_path, model_name, do_validation)
                - where model_name is a string of the form: model_name = "SVM-Classifier(%2d)_%3d_%1d_%2d_%1d" % (max_stream_count, n_cmpts, bs, ws, to)
                - do_validation flag checks for .mseed's existence in a labeled PNG directory (set to 0 in realtime_processing)

        - mass_mseed_classify (specify correct & "mmmeh" thresholds and max .mseeds from each class to classify)
            - calls mass_mseed_classification
                1) Loads .mseeds from *class*_validation directory
                2) Extracts timestamp from these .mseeds and locates corresponding raw .mseed
                3) Evaluates single .mseed prediction via:
                    a) svm_classify_stream
                        i) process_stream_to_pca -> data_to_predict
                            1) performs pca on each useful_channel of a given stream and returns single conjoined trace
                        ii) svm_predict(data_to_predict) -> returns list of 2-element tuples containing class label and probability 
                    b) prediction scrutinization:
                        i) if top class prediction is greater than "correct" threshold, increment 0th index of mseed_pred[]
                        ii) else if ^ is greater than "mmmeh" threshold increment 1st index of mseed_pred[]
                        iii) otherwise increment 2nd index of mseed_pred (misclassified)
                4) Tally all mseed predictions for a given class and:
                    a) output matrix containing sum of that class' mseed_pred[] scores (total & acceptability score from 0th & 1st elements)
                    b) output matrix deconstructing the breakdown of the misclassified .mseeds for a particular class
'''
labeled_png_path        = "D:\\labeled_data\\png\\"
labeled_data_path       = "D:\\labeled_data\\optimized_and_labeled_triggers\\"
classifier_location     = "C:\\Users\\David\\Documents\\SVM\\models\\"
unlabeled_triggers_path = "D:\\trigger\\"
model_performance_path  = "C:\\Users\\David\\Documents\\SVM\\model_metrics\\model_performance.txt"
model_metrics_path      = "C:\\Users\\David\\Documents\\SVM\\model_metrics\\"

grid_search_params_path = 'C:\\Users\\David\\Documents\\SVM\\svm_grid_search\\svm_grid_search_params.txt'
confusion_matrix_path   = 'C:\\Users\David\\Documents\\SVM\\SVM_confusion_matrices\\'

class_dict              = {'MEQ' : 0, 'CASSM' : 1, 'DRILLING' : 2, 'ERT' : 3}

useful_channels         = [(2, 'PDB03')]
useful_channels         += [(54, 'OT16Z')]
useful_channels         += [(55, 'OT16X')]
max_stream_count        = 3327                                                  # max streams to consider for training
total_data              = []                                                    # Contains lists of channels for all streams from all classes
class_labels            = []
expected_data_size      = 0
padding_cnt             = 0

plot_wiggles            = 0                                                     # Debugging: whether to show wiggle plots at each stage
sequentially_load_pngs  = 0

def main():
    global expected_data_size
    global total_data
    global class_labels
        
    window_size      = 1700
    trigger_offset   = 300
    block_size       = 2
    n_cmpts          = floor(window_size / block_size) - 1

    # Grid search for PCA params
    window_sizes     = [1500, 1700, 1900]
    block_sizes      = [2]
    trigger_offsets  = [200, 300, 400]

    do_train                = 0
    do_svm_grid_search      = 0
    iter_pca_params         = 0
    classify_single_mseed   = 0
    mass_mseed_classify     = 1
    
    # Mass .mseed classification params
    correct_threshold       = 0.70                                              # predictions above this are considered confidently correct
    mmmeh_threshold         = 0.51                                              # predictions below ^ and above this are ostensibly correct               
    max_classify_cnt        = 3327                                              # max. amt. of .mseeds to classify per class (subject to avail.)

    model_name = "SVM-Classifier(%2d)_%3d_%1d_%2d_%1d" % \
                 (max_stream_count, n_cmpts, block_size, window_size, trigger_offset)

    classes = ['ert','meq', 'cassm', 'drilling']
    #model_path = "{0}{1}-{2}".format(classifier_location, channel_name, model_name)

    if (do_train):
        expected_data_size = 0
        if (iter_pca_params == 1):
            for ws in window_sizes:
                    for bs in block_sizes:
                            for to in trigger_offsets:
                                n_cmpts          = floor(ws / bs) - 1
                                model_name = "SVM-Classifier(%2d)_%3d_%1d_%2d_%1d" % (max_stream_count, n_cmpts, bs, ws, to)
                                expected_data_size = 0
                                total_data   = []
                                class_labels = []
                                tf = TrainingFeatures(ws, to, bs)
                                train_with_these_features(tf, do_svm_grid_search)

                                if mass_mseed_classify:
                                    mass_mseed_classification(correct_threshold, mmmeh_threshold, max_classify_cnt, model_name, classes)

        else:
            expected_data_size = 0
            total_data   = []
            class_labels = []
            tf = TrainingFeatures(window_size, trigger_offset, block_size)
            train_with_these_features(tf, do_svm_grid_search)

    elif classify_single_mseed:
        #stream_path = "D:\\trigger\\1543378742.86.mseed"                # CASSM
        #stream_path = "D:\\trigger\\1544197621.54.mseed"               # Drilling
        stream_path = "D:\\trigger\\1544136427.33.mseed"               # MEQ
        #stream_path = "D:\\trigger\\1543366796.60.mseed"               # ERT
        #stream_path = "D:\\trigger\\1544203603.19.mseed"
        #stream_path = "D:\\trigger\\1543365081.20.mseed"
        classify_mseed(stream_path, model_name, do_validation=1)


    # Iterates through each useful channel for every class and prints respective model score
    elif (mass_mseed_classify):
        if (iter_pca_params == 1):
            for ws in window_sizes:
                    for bs in block_sizes:
                            for to in trigger_offsets:
                                n_cmpts          = floor(ws / bs) - 1
                                model_name =  "SVM-Classifier(%2d)_%3d_%1d_%2d_%1d" % \
                                              (max_stream_count, n_cmpts, bs, ws, to)
                                mass_mseed_classification(correct_threshold, mmmeh_threshold, max_classify_cnt, model_name, classes)
        else:
            mass_mseed_classification(correct_threshold, mmmeh_threshold, max_classify_cnt, model_name, classes)


class TrainingFeatures:
    window_size      = 0
    trigger_offset   = 0
    block_size       = 0
    pca_n_components = 0

    def __init__(self, ws, to, bs):
        self.window_size        = ws
        self.trigger_offset     = to
        self.block_size         = bs
        self.pca_n_components   = floor(ws / bs) - 1

    def return_features(self):
        return self.window_size, self.trigger_offset, self.block_size, self.pca_n_components

    def return_info(self):
        return "\nWindow size: {0}\nTrigger offset: {1}\nBlock size: {2}\n# components: {3}".format(
            self.window_size, self.trigger_offset, self.block_size, self.pca_n_components)

def create_classifier(do_grid_search,tf):
    channel_data   = []
    window_size, trigger_offset, block_size, pca_n_components = tf.return_features()

    for t, trace in enumerate(total_data):                                      # extracts relevant channel pca's for training
        #channel_data.append(total_data[t][channel_index])
        channel_data.extend(trace)
    
    
    X_train, X_test, y_train, y_test = train_test_split(total_data, class_labels,
                                                        test_size=0.3)

    print("training vs. testing split\n")
    X_train_2D = transform3Dto2D(np.array(X_train))
    X_test_2D = transform3Dto2D(np.array(X_test))

    classifier = svm.SVC(C=20, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf',
                max_iter=-1, probability=True, random_state=None, shrinking=True,
                tol=0.001, verbose=False)

    # the magic:
    classifier.fit(X_train_2D, y_train)
    print("%0.04f%% of data used (%d/%d)" % ((len(total_data)/expected_data_size)*100, len(total_data), expected_data_size))
    print("Performing SVM grid search using PCA params {ws: %d, to: %d, bs: %d}\n" % (window_size, trigger_offset, block_size))
    if do_grid_search:
        params = [{'C': [1, 10, 20, 30], 'kernel':['rbf'],
                  'gamma': [1, 5, 10], 'degree': [3], 'decision_function_shape': ['ovr']}]
        #params = [{'C': [10], 'kernel':['rbf'], 'gamma': [0.85, 0.9, 0.92]}]
        grid_search = GridSearchCV(estimator=classifier, param_grid=params, scoring='accuracy', cv=10, n_jobs=-1, verbose=10)
        grid_search = grid_search.fit(X_train_2D, y_train)
        best_acc = grid_search.best_score_
        best_params = grid_search.best_params_
        ts = calendar.timegm(time.gmtime())
        stream  = '[%s] highest accuracy of %.06f found with following SVM params: %s\n on these PCA params: (ws: %d, to: %d, bs: %d)' % \
                  (ts, best_acc, best_params, window_size, trigger_offset, block_size)
        file_name = grid_search_params_path
        
        with open(file_name, 'a+') as the_file:
                the_file.write(stream)

    return X_train_2D, X_test_2D, y_train, y_test, classifier

def train_with_these_features(tf, do_svm_grid_search):
    window_size, trigger_offset, block_size, pca_n_components = tf.return_features()
    global total_data
    global class_labels
    global expected_data_size

    total_data = []
    class_labels = []
    optimized_mseed_class_path = labeled_data_path 

    ert_stream_path      = optimized_mseed_class_path + 'training_mseeds\\ert_training\\*.mseed'
    drilling_stream_path = optimized_mseed_class_path + 'training_mseeds\\drilling_training\\*.mseed'
    cassm_stream_path    = optimized_mseed_class_path + 'training_mseeds\\cassm_training\\*.mseed'
    meq_stream_path      = optimized_mseed_class_path + 'training_mseeds\\meq_training\\*.mseed'

    build_data_set(cassm_stream_path,    max_stream_count, tf)              # Run through directories of optimized .mseed files and
    build_data_set(drilling_stream_path, max_stream_count, tf)              # append traces to total_data and class type to class_labels 
    build_data_set(meq_stream_path,      max_stream_count, tf)
    build_data_set(ert_stream_path,      max_stream_count, tf)

    X_train, X_test, y_train, y_test, classifier = create_classifier(do_svm_grid_search, tf)
    model_evaluate(X_train, X_test, y_train,
                    y_test, classifier, tf, max_stream_count, show_matrix=1, save_eval=1)
    
    classifier_name = "SVM-Classifier(%d)_%d_%d_%d_%d" % (max_stream_count, pca_n_components,
                                                            block_size, window_size, trigger_offset)

    save_classifier(classifier, classifier_location + classifier_name)

# Given a directory for a class' labeled .mseed files, perform PCA on stream's
# useful_channels and append the extended tuple to total_data and extract class_label to
# append to class_labels
def build_data_set(labeled_class_path, stream_count, tf):
    global expected_data_size
    global padding_cnt

    window_size, trigger_offset, block_size, pca_n_components = tf.return_features()
    all_stream_file_names   = sorted(glob.glob(labeled_class_path))
    if not sequentially_load_pngs:
        random.shuffle(all_stream_file_names)

    total_streams           = len(all_stream_file_names) if (stream_count == -1 or stream_count > len(all_stream_file_names)) else stream_count
    expected_data_size      += total_streams
    print("Loading {0} streams & finding principal components on {1} channel(s) from {2}".format(
           total_streams, len(useful_channels), labeled_class_path))

    plots_shown = 0
    for s in trange(total_streams, leave=True):
        trace_tuple             = []                                            # list of traces for each useful channel
        channel_triggers        = []                                            # list of respective triggers for traces
        stream_pca              = []                                            # pca-ified version of trigger-based trace window
        useable_stream          = 1
        st = read(all_stream_file_names[s])                                     # Loads optimized stream

        # Iterate through each uc and feed the pca-ified wiggles into 
        # stream_pca trace
        for uc in range(len(useful_channels)):
            channel_name = useful_channels[uc][1]
            
            t = st[uc]
            trace_tuple.append(t)                                               # loads trace from optimized stream (0th is PDB11, 1st is OT16, 2nd to last=triggers, last=class_label)
            channel_triggers.append(st[len(st) - 2].data[uc])                   # respective triggers always in 2nd to last index
        
        for t, trace in enumerate(trace_tuple):                                 # extracts window based on trigger and finds pca
            pca_wiggle = []
            #trigger_start = channel_triggers[t]
            trigger_stream = return_trigger_index(trace, useful_channels[t][1], -1)
            trigger_start  = trigger_stream[0].data[0]
            trace_start   = trigger_start - trigger_offset
            trace_end = trace_start + window_size

            if (trace_start <= 0):
                useable_stream = 0                                              # trigger is too early, window is likely erroneous
                trace_start = 0                                                             
            else:
                last_data_pt = trace[len(trace)-1]
                trace_window    = []
                if (len(trace) < trace_end):                                # trace window extends beyond size of total trace
                    trace_window = np.pad(trace[trace_start:len(trace)], (0, (trace_end - len(trace))),
                                          'constant', constant_values=last_data_pt)
                    #useable_stream = 0
                    padding_cnt += 1
                else:
                    trace_window = trace[trace_start:trace_end]
                pca, pca_wiggle, reconstructed_wiggle = pca_reduce(trace_window, pca_n_components, block_size)
            
            if plot_wiggles:
                #plot_wiggle(pca_wiggle)
                plt.plot(pca_wiggle)
                plots_shown += 1
                print("have %d wiggles plotted" % plots_shown)
            
            # conjoin pca's of each channel
            stream_pca.extend(pca_wiggle) 

        if (useable_stream):
            if plot_wiggles:
                #for stream in stream
                plt.plot(stream_pca)
                plt.show()
                plt.clf()
                plt.cla()
                plt.close()
            total_data.append(stream_pca)
            class_label = st[len(st) - 1].data[0]                                   # Class label always in last index
            class_labels.append(class_label)

    print("PCA performed on %d out of %d streams (%.03f%%), %d streams with padding" %
     (len(total_data), expected_data_size, 100*(len(total_data)/expected_data_size), padding_cnt))

# serializes model for simple storage & retrieval
def save_classifier(classifier, save_path):
        try:
                print("Attempting to save classifier in " + save_path)
                joblib.dump(classifier, save_path)
        except Exception as e:
                print("Could not save model in " + save_path + "[" + str(e) + "]")

# Finds model metrics (accuracy) on channel model
def model_evaluate(X_train, X_test, y_train, y_test, classifier, training_features, max_stream_count, show_matrix, save_eval):
    
    window_size, trigger_offset, block_size, pca_n_components = training_features.return_features()    

    classifier_name = "SVM-Classifier(%d)_%d_%d_%d_%d" % (max_stream_count, pca_n_components, block_size, window_size, trigger_offset)
    print("\n*** Evaluating model performance using " + str(len(useful_channels)) + " pca-ified channels per Stream")

    y_test_prediction   = classifier.predict(X_test)
    y_train_prediction  = classifier.predict(X_train)

    # model metrics
    precision    = metrics.precision_score(y_test, y_test_prediction, average='micro')
    recall       = metrics.recall_score(y_test, y_test_prediction, average='micro')
    F1           = 2 * ((precision * recall ) / (precision + recall))
    testing_acc  = metrics.accuracy_score(y_test, y_test_prediction)
    training_acc = metrics.accuracy_score(y_train, y_train_prediction)

    print("Precision: %.04f\nRecall: %0.04f\nF1 = %.03f" % (precision, recall, F1))
    print("Accuracy on testing data: %.06f" % testing_acc)  
    print("Accuracy on training data: %.06f" % training_acc)

    plot_confusion_matrix(y_test, y_test_prediction, classes=class_labels, normalize=True, title=classifier_name)
    if (show_matrix):
        try:
            #plt.show()
            #plt.clf()
            plt.savefig(confusion_matrix_path + classifier_name)
        except Exception as e:
            print('could not save matrix [' + ']')
        
    if (save_eval):
            ts = calendar.timegm(time.gmtime())
            date = ts2date(ts, 0)
            date = date[4:8]
            date = date[:2] + "-" + date[2:]
            cm_train = confusion_matrix(y_train, y_train_prediction)
            cm_train = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]
            cm_test = confusion_matrix(y_test, y_test_prediction)
            cm_test = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis] #normalized

            stream = '\n[%s] (%.04f)_Testing (%.04f)_Training acc. n_comp: %4d, block_size: %d, window: %d, initial_offset: %d\n' % \
                (ts, testing_acc, training_acc, pca_n_components, block_size, window_size, trigger_offset)
            print(stream)
            model_metrics_file_location = model_metrics_path + "\\" + date + "_model-metrics.txt"
            try:
                    print('Saving model metrics & parameters in ' + model_metrics_file_location)
                    with open(model_metrics_file_location, 'a') as f:
                            f.write(stream)
            except Exception as e:
                    print("could not open " + model_metrics_file_location + "[" + e + "]")


def mass_mseed_classification(correct_pred_thresh, mmmeh_threshold, max_classify_cnt, model_name, classes):
    
    total_class_predictions        = np.zeros(shape=(len(classes),  3))                             # Performance of all classes correct, maybe correct, misclassified count
    total_class_misclassifications = np.zeros(shape=(len(classes), len(classes)))                   # Distribution of misclassified .mseeds for each class
    model_path = "{0}{1}".format(classifier_location, model_name)
    model = joblib.load(model_path)

    for class_index, class_type in enumerate(classes):
            skipped_mseeds              = 0
            class_misclassifications    = np.zeros(len(classes))
            optimized_mseed_class_path  = labeled_data_path
            optimized_mseed_class_path  += "validation_mseeds\\" + class_type + "_testing\\*.mseed"
            
            if sequentially_load_pngs:
                all_class_mseeds = sorted(glob.glob(optimized_mseed_class_path))
            else:
                all_class_mseeds = glob.glob(optimized_mseed_class_path)
                random.shuffle(all_class_mseeds)

            num_class_mseeds = max_classify_cnt if (len(all_class_mseeds) > max_classify_cnt) else len(all_class_mseeds)
            
            print("Performing mass " + class_type.upper() + " .mseed classification on %d streams from %s. " % (num_class_mseeds, optimized_mseed_class_path))
            for mseed in trange(num_class_mseeds, leave=True):
                timestamp = re.search('[0-9]+(\.[0-9][0-9]?)?', all_class_mseeds[mseed]).group(0)   # extracts timestamp from file name
                mseed_pred          = np.zeros(3, dtype=np.int32)                                   # single mseed prediction result [correct, maybe correct, misclassified count]
                which_wrong_class   = np.zeros(shape=(len(classes)))                                # specifies which was the wrong class prediction
                unlabeled_mseed_path = unlabeled_triggers_path + timestamp + ".mseed"
                try:
                        st = read(unlabeled_mseed_path)
                except Exception as e:
                        skipped_mseeds += 1
                        #print("Could not read stream at " + str(unlabeled_mseed_path) + "[" + str(e) + "]")
                        continue

                predictions = svm_classify_stream(st, model, model_name, class_type=-1)

                if (predictions[0][0] == class_type.upper()) and \
                    (float(predictions[0][1]) > correct_pred_thresh):           # Confident & correct prediction
                        mseed_pred[0] +=1

                elif (predictions[0][0] == class_type.upper()) and \
                    (float(predictions[0][1]) > mmmeh_threshold):               # Correct classification, somewhat confident
                        mseed_pred[1] += 1
                else:                                                           # Misclassified or timidly confident
                    mseed_pred[2] += 1
                    class_key = class_dict.get(predictions[0][0])
                    class_misclassifications[class_key] += 1
                total_class_predictions[class_index] += mseed_pred
            total_class_misclassifications[class_index] = class_misclassifications

    # hacky way to easily save console output to text file across multiple functions
    stdoutOrigin = sys.stdout
    sys.stdout = open(model_performance_path, "a+")
    print("\n\nModel performance on [%s]\nConfident Threshold: %0.02f; mmmeh threshold: %.02f" % (model_path, correct_pred_thresh, mmmeh_threshold))
    print_classification_matrix(total_class_predictions, classes)
    print_misclassification_breakdown(total_class_misclassifications)
    sys.stdout.close()
    sys.stdout = stdoutOrigin

# Given a stream, return list of tuples with probabilities for each class
def svm_classify_stream(stream, model, model_name, class_type):
        pca_n_components, pca_block_size, window_size, trigger_offset = load_params(model_name)
        tf               = TrainingFeatures(window_size, trigger_offset, pca_block_size)

        data_to_predict  = process_stream_to_pca(stream, tf, class_type)

        return svm_predict(data_to_predict, model)
        #return svm_predict(data_to_predict, model)

# PCA parameters are extracted from the classifier's file name
# e.g. from "SVM-Classifier(500)_449_2_900_200" returns 449, 2, 900, 200 (skips max stream count)
# Order is: max_stream_count, n_pca_components, block_size, window_size, trigger_offset
def load_params(classifier_location):
        model_name       = classifier_location.split("\\")[classifier_location.count("\\")]
        param_chunk      = model_name.split("SVM")[1]
        pca_params       = re.findall(r'\d+', param_chunk)                      # returns numbers from string
        return int(pca_params[1]), int(pca_params[2]), int(pca_params[3]), int(pca_params[4])

# Given a stream, TrainingFeatures object, and the indices of useful_channels,
# return a 1D list with extended len(useful_channels) conjoined of pca-ified wiggles
def process_stream_to_pca(stream, tf, class_type):
    multi_channel_pca_wiggle = []

    for uc in useful_channels:
        channel_name = uc[1]
        window_size, trigger_offset, block_size, pca_n_components = tf.return_features()

        ## ! ##
        ## ! ## Important: this trace preprocessing must be replicated as in stream_optimization_labelling.py
        ## ! ##
        useful_channel  = stream[uc[0]].copy()
        if 'OT' in channel_name:
            useful_channel.filter('lowpass', freq=6000)
        elif 'PDB' in channel_name:
            useful_channel.filter('lowpass', freq=3000)

        trace_start = 300
        useful_channel.data = useful_channel.data[trace_start:]
        useful_channel.data = trace_tail_chopper(useful_channel.data)

        trigger_stream  = return_trigger_index(useful_channel, channel_name, class_type)
        trigger_start   = int(trigger_stream[0].data[0])
        trace_start     = trigger_start - trigger_offset if ((trigger_start - trigger_offset) > 0) else 0
        trace_end       = trace_start + window_size

        last_data_pt    = useful_channel[len(useful_channel)-1]
        trace_window    = []

        # all wiggles must be the same size when entering SVM
        if (len(useful_channel) < trace_end):
            trace_window = np.pad(useful_channel[trace_start:len(useful_channel)],
                                    (0, (trace_end - len(useful_channel))),
                                    'constant', constant_values=last_data_pt)
        else:
            trace_window = useful_channel[trace_start:trace_end]

        pca, pca_wiggle, reconstructed_wiggle = pca_reduce(trace_window, pca_n_components, block_size)

        multi_channel_pca_wiggle.extend(pca_wiggle.flatten())
    multi_channel_pca_wiggle = np.array(multi_channel_pca_wiggle).reshape(1, -1)

    return multi_channel_pca_wiggle

# Given a 1D list of pca-ified wiggle(s), unleash provided model on data
def svm_predict(data_to_predict, model):
    data_predictions = model.predict_proba(data_to_predict)                     # magic function to return prediction probabilities of supplied model on data
    predictions      = []
    class_prediction = []

    for key, value in class_dict.items():
            class_prediction = (key, '%0.04f' % data_predictions[0][value])
            predictions.append(class_prediction)

    return sorted(predictions,key=lambda x: x[1], reverse=True)                 # return class probabilities in descending order

# Given an .mseed timestamp and its class, check to see whether this .mseed 
# exists in a directory (likely one used for training) to classify on novel data
def check_mseed_existence(timestamp, mseed_dir):
    for root, dirs, mseeds in os.walk(mseed_dir):
        mseed_png_name = timestamp + "-L.mseed"
        if mseed_png_name in mseeds:
            return 1
    return 0

# from: https://tinyurl.com/yawodqj5
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.YlOrBr):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes_values = unique_labels(classes)
    classes_labels = find_class_label_from_value(classes_values, 0)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes_labels, yticklabels=classes_labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')


    plt.rc('font', size=20)          # controls default text sizes
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# Returns the corresponding class given the key of the class_dict (bit hacky since this is not the purpose of dictionaries)
def find_class_label_from_value(class_values, single_class):
        classes_labels = []
        if single_class:
            return list(class_dict.keys())[list(class_dict.values()).index(class_values)]
        else:
            for cv in class_values:
                    classes_labels.append(list(class_dict.keys())[list(class_dict.values()).index(cv)])
            return classes_labels

# Utility function to morph data into useable form
# eg reduce (6, 16, 4) to (6,32)
def transform3Dto2D(np_array_3D):
        print(np.shape(np_array_3D))                                            
        np_array_2D = np.zeros(shape=(np.shape(np_array_3D)[0],np.shape(np_array_3D)[1]*np.shape(np_array_3D)[2]))
        array_2D = []
        for i in range((np.shape(np_array_2D)[0])):
                new_row = []
                for j in range((np.shape(np_array_3D)[1])):
                        new_row.extend(np_array_3D[i][j])
                array_2D.append(new_row)
        return np.array(array_2D).tolist()

# Given a path to a Stream and the model_name, find the .mseed's probability prediction breakdown across classes
def classify_mseed(stream_path, model_name, do_validation):
    st = read(stream_path)
    
    if plot_wiggles:
        st[2:50].plot(equal_scale=False)

    if (plot_wiggles):
        for uc in useful_channels:
            st[uc[0]].plot(method='full')

    model_path = "{0}{1}".format(classifier_location, model_name)
    model = joblib.load(model_path)
    # Prediction returns a 4x(2x1) list containing each class type and
    # the probability of its correct classification
    
    start_time = timer()
    predictions = svm_classify_stream(st, model, model_name, class_type=-1)
    end_time = timer()
    print("Single .mseed classification took %0.05fs" % (end_time - start_time))
    timestamp = re.search('[0-9]+(\.[0-9][0-9]?)?', stream_path).group(0)

    if do_validation:
        if is_correct_prediction(timestamp, predictions[0][0]):                     # Validates prediction
            print("\n (!) Correctly predicted as %s" % predictions[0][0])
        else:
            actual_class_type = find_mseed_class(timestamp)
            print("A misclassification! %s.mseed is actually %s" % (timestamp, actual_class_type.upper()))

    for prediction in predictions:
        print("%-8s %0.03f%%" % (prediction[0], float(prediction[1])*100), end="\n")


# Given the name of an unlabeled .mseed and its prediction, return whether .mseed exists in predicted class' png folder
# e.g. ('1543365023.20', 1) -> 1    
def is_correct_prediction(mseed_name, class_prediction):
    labeled_png_path = labeled_png_path + class_prediction
    for root, dirs, files in os.walk(labeled_png_path):
        mseed_png_name = mseed_name + ".png"
        if mseed_png_name in files:
            return 1
    return 0

# Given the timestamp name of an .mseed, look through all png folders and return
# matching class
def find_mseed_class(timestamp):
    classes = ['meq', 'cassm', 'drilling', 'ert']

    for class_type in classes:
        labeled_png_path = labeled_png_path + class_type
        for root, dirs, files in os.walk(labeled_png_path):
            mseed_png_name = timestamp + ".png"
            if mseed_png_name in files:
                return class_type

    return 'This .mseed is without a home.'

# Utility function to cleanly print out misclassified distribution of a class in ascending order
def print_misclassification_breakdown(total_class_misclassifications):
    print("\nMisclassification breakdown")
    for cl, class_breakdown in enumerate(total_class_misclassifications):
        class_label = find_class_label_from_value(cl, single_class=1)
        print("\n%8s) " % class_label, end="")
        total_misclassified = sum(class_breakdown)
        class_misclassifications = []
        for wc, wrong_class in enumerate(class_breakdown):
            wrong_class_label = find_class_label_from_value(wc, single_class=1)
            if total_misclassified == 0:
                wrong_class_proportion = 0
            else:
                wrong_class_proportion = 100 * wrong_class / total_misclassified
            wrong_class_tuple = (wrong_class_label, wrong_class_proportion)
            class_misclassifications.append(wrong_class_tuple)

        class_misclassifications.sort(key=lambda x: x[1])
        for cm in class_misclassifications:
            print("{:>8s}: {:02.02f}%".format(cm[0], cm[1]), end=" ")

# Cleanly prints contents of a 4 x (1 x 3) list containing the class prediction (correct, somewhat correct, misclassified)
# for all classes
def print_classification_matrix(total_class_predictions, classes):
        print("\t   ", end="")
        for ch, channel in enumerate(useful_channels):
            print("%6s" % channel[1].ljust(6), end="+ ")
        print("Total   Acceptability", end="\n\t  ")
        for uc in range(len(useful_channels)+1):
            print("-----------", end="")
        print("")
        #print("".join(["--------------" for uc in range(useful_channels+1)]))
        for cp, class_prediction in enumerate(total_class_predictions):
            title_stream = "{:9s}".format(classes[cp].rjust(9).upper())
            print(title_stream, end="[")

            channel_stream = ""
            for p in class_prediction:
                channel_stream += "%4d " % p
            else:
                print(channel_stream, end="")
            print("] E=%0*d" % (len(str(max_stream_count)), sum(class_prediction)), end=" ")
            print_class_performance(class_prediction, classes)
            print("")

# Returns tuple with size of useful_channels with acceptability scores (sum of
# correct + kinda_correct predictions) for each channel
def print_class_performance(class_prediction, classes):
    class_scores = []

    stream_count                = sum(class_prediction)
    confidently_predicted       = class_prediction[0]                     # count of valid predictions above arbitrary confidence threshold (usually 70-80%+)
    kinda_confidently_predicted = class_prediction[1]                     # valid predictions below ^ and above a different threshold (usually 40-60%)
    misclassified               = class_prediction[2]
    
    correct_pred_score          = confidently_predicted       / stream_count      # ratios
    kinda_correct_pred_score    = kinda_confidently_predicted / stream_count
    misclassified_score         = misclassified               / stream_count

    acceptability_score         = correct_pred_score + kinda_correct_pred_score
    class_scores.append("%.02f" % acceptability_score)

    print("( ", end="")
    for cs in class_scores:
        print("%3s " % cs, end="")
    print(")", end="")

if __name__ == '__main__':
    main()
