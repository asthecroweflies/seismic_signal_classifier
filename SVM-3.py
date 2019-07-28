'''

    SVM Classifier 3000
    

  Trains a support vector machine on multiple 1D arrays of 
  conjoined PCA-ified wiggles.  
  Also divides the data into internal testing / training sets 
'''
labeled_data_path       = "D:\\labeled_data\\optimized_and_labeled_triggers\\"

grid_search_params_path = 'C:\\Users\\David\\Documents\\SVM\\svm_grid_search\\svm_grid_search_params.txt'
confusion_matrix_path   = 'C:\\Users\David\\Documents\\SVM\\SVM_confusion_matrices\\'

def create_classifier(do_grid_search):
    channel_data   = []

    for t, trace in enumerate(total_data):                                      # extracts relevant channel pca's for training
        #channel_data.append(total_data[t][channel_index])
        channel_data.extend(trace)
    
    
    X_train, X_test, y_train, y_test = train_test_split(channel_data, class_labels,
                                                        test_size=0.2)

    print("training vs. testing split\n")
    X_train_2D = transform3Dto2D(np.array(X_train))
    X_test_2D = transform3Dto2D(np.array(X_test))

    classifier = svm.SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovo', degree=3, gamma=10, kernel='rbf',
                max_iter=-1, probability=True, random_state=None, shrinking=True,
                tol=0.001, verbose=False)

    # the magic:
    classifier.fit(X_train_2D, y_train)
    print("%0.04f%% of data used (%d/%d)" % ((len(total_data)/expected_data_size)*100, len(total_data), expected_data_size))
    
    if do_grid_search:
        params = [{'C': [10, 20, 30], 'kernel':['rbf'],
                  'gamma': [1, 5, 10], 'degree': [3], 'decision_function_shape': ['ovr', 'ovo']}]
        #params = [{'C': [10], 'kernel':['rbf'], 'gamma': [0.85, 0.9, 0.92]}]
        grid_search = GridSearchCV(estimator=classifier, param_grid=params, scoring='accuracy', cv=10, n_jobs=4, verbose=10)
        grid_search = grid_search.fit(X_train_2D, y_train)
        best_acc = grid_search.best_score_
        best_params = grid_search.best_params_
        ts = calendar.timegm(time.gmtime())
        stream  = '[%s] %s) highest accuracy of %.06f found with following params: %s\n' % (ts, str(channel_index), best_acc, best_params)
        filename = grid_search_params_path
        
        with open(file_name, 'a+') as the_file:
                the_file.write(stream)

    return X_train_2D, X_test_2D, y_train, y_test, classifier


def train_these_features(tf, do_svm_grid_search):
    window_size, trigger_offset, block_size, pca_n_components = tf.return_features()
    global total_data
    global class_labels
    global expected_data_size

    for c, uc in enumerate(useful_channels):                                    # create classifier for each channel
        channel_name = str(uc[1])
        total_data = []
        class_labels = []
        optimized_mseed_class_path = labeled_data_path 

        ert_stream_path      = optimized_mseed_class_path + 'ert_training\\*.mseed'
        drilling_stream_path = optimized_mseed_class_path + 'drilling_training\\*.mseed'
        cassm_stream_path    = optimized_mseed_class_path + 'cassm_training\\*.mseed'
        meq_stream_path      = optimized_mseed_class_path + 'meq_training\\*.mseed'

        build_data_set(cassm_stream_path,    max_stream_count, tf)              # Run through directories of optimized .mseed files and
        build_data_set(drilling_stream_path, max_stream_count, tf)              # append traces to total_data and class type to class_labels 
        build_data_set(meq_stream_path,      max_stream_count, tf)
        build_data_set(ert_stream_path,      max_stream_count, tf)

        X_train, X_test, y_train, y_test, classifier = create_classifier(c, do_svm_grid_search)
        model_evaluate(c, X_train, X_test, y_train,
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
            trigger_start = channel_triggers[t]
            trace_start   = trigger_start - trigger_offset
            trace_end = trace_start + window_size

            if (trace_start <= 0) and (is_fft == 0):
                useable_stream = 0                                              # trigger is too early, remove to prevent bias
                trace_start = 0                                                             
            else:
                last_data_pt = trace[len(trace)-1]
                trace_window    = []
                if (trace_end - len(trace) < 400):#(len(trace) < trace_end):
                    trace_window = np.pad(trace[trace_start:len(trace)], (0, (trace_end - len(trace))),
                                          'constant', constant_values=last_data_pt)
                    padding_cnt += 1
                else:
                    trace_window = trace[trace_start:trace_end]
                pca, pca_wiggle, reconstructed_wiggle = pca_reduce(trace_window, pca_n_components, block_size)
            
            if (plots_shown < 15):
                #plot_wiggle(pca_wiggle)# why cassms look funny
                plots_shown += 1
            
            #stream_pca.append(pca_wiggle)
            stream_pca.extend(pca_wiggle) # continuous wiggle of all pca-ified wiggles
        if (useable_stream):
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

    classifier_name = "%SVM-Classifier(%d)_%d_%d_%d_%d" % (max_stream_count, pca_n_components, block_size, window_size, trigger_offset)
    print("\n*** Evaluating model performance on wiggles with " + len(useful_channels) + "channel pca's")

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

            stream = '[%s - %s] (%.04f)_Testing (%.04f)_Training acc. n_comp: %4d, block_size: %d, window: %d, initial_offset: %d\n' % \
                (ts, channel_name.rjust(6), testing_acc, training_acc, pca_n_components, block_size, window_size, trigger_offset)
            print(stream)
            model_metrics_file_location = model_metrics_path + "\\" + date + "_model-metrics.txt"
            try:
                    print('Saving model metrics & parameters in ' + model_metrics_file_location)
                    with open(model_metrics_file_location, 'a') as f:
                            f.write(stream)
            except Exception as e:
                    print("could not open " + model_metrics_file_location + "[" + e + "]")

def mass_mseed_classification(correct_pred_thresh, mmmeh_threshold, max_classify_cnt, model_name, classes):
    
    total_class_predictions        = np.zeros(shape=(len(classes),  3))                       # Performance of all classes correct, maybe correct, misclassified count
    total_class_misclassifications = np.zeros(shape=(len(classes), shape=(len(classes))))     # Distribution of misclassified .mseeds for each class

    for class_index, class_type in enumerate(classes):
            skipped_mseeds                   = 0

            class_misclassifications = np.zeros(shape=(len(useful_channels), len(classes)))
            #class_predictions        = np.zeros(shape=(len(classes), 3))
            optimized_mseed_class_path = D_class_labeled_data_path
            optimized_mseed_class_path += class_type + "_validation\\*.mseed"
            
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
                model_path = "{0}-{1}".format(classifier_location, model_name)
                unlabeled_mseed_path = unlabeled_triggers_path + timestamp + ".mseed"
                try:
                        st = read(unlabeled_mseed_path)
                except Exception as e:
                        skipped_mseeds += 1
                        #print("Could not read stream at " + str(unlabeled_mseed_path) + "[" + str(e) + "]")
                        continue

                predictions = svm_classify_stream(st, model_path, class_type=-1)

                if (predictions[0][0] == class_type.upper()) and \
                    (float(predictions[0][1]) > correct_pred_thresh):       # Confident & correct prediction
                        mseed_pred[0] +=1

                elif (predictions[0][0] == class_type.upper()) and \
                    (float(predictions[0][1]) > mmmeh_threshold):           # Correct classification, somewhat confident
                        mseed_pred[1] += 1
                else:                                                       # Misclassified or timidly confident
                    mseed_pred[2] += 1
                    class_key = class_dict.get(predictions[0][0])
                    class_misclassifications[class_key] += 1

            total_class_misclassifications[class_index] = class_misclassifications
            total_class_predictions[class_index]        = mseed_pred

        stdoutOrigin = sys.stdout
        sys.stdout = open(model_performance_path, "a+")
        print("\n((*************** | ((***************))\nModel performance on [%s]" % model_path)
        print_classification_matrix(total_class_predictions, classes)
        print_misclassification_breakdown(total_class_misclassifications)
        sys.stdout.close()
        sys.stdout = stdoutOrigin

# Given a stream, return list of tuples with probabilities for each class
def svm_classify_stream(stream, best_classifier_location, class_type):
        model            = joblib.load(best_classifier_location)
        pca_n_components, pca_block_size, window_size, trigger_offset = load_params(best_classifier_location)
        tf               = TrainingFeatures(window_size, trigger_offset, pca_block_size)
        
        for uc in useful_channels:
            useful_channel_indices.append(uc[0])

        data_to_predict  = process_stream_to_pca(stream, tf, useful_channel_indices, class_type).flatten()

        return svm_predict(data_to_predict.reshape(1, -1), model, channel_name)

# PCA parameters are extracted from the classifier's file name
# e.g. from "SVM-Classifier(500)_449_2_900_200" returns 449, 2, 900, 200 (skips max stream count)
# Order is: max_stream_count, n_pca_components, block_size, window_size, trigger_offset
def load_params(classifier_location):
        model_name       = classifier_location.split("\\")[classifier_location.count("\\")]
        param_chunk      = model_name.split("SVM")[1]
        pca_params       = re.findall(r'\d+', param_chunk)                       # returns numbers from string
        return int(pca_params[1]), int(pca_params[2]), int(pca_params[3]), int(pca_params[4])

# Given a stream, TrainingFeatures object, and the useful_channel index,
# return a 1D list with extended pca-ified wiggles of len(useful_channels)
def process_stream_to_pca(stream, tf, channel_indices, class_type):
    multi_channel_pca_wiggle = []

    for ci in channel_indices:
        channel_name = useful_channels[ci][1]
        window_size, trigger_offset, block_size, pca_n_components = tf.return_features()

        #### Important: this trace preprocessing must be replicated as in stream_optimization_labelling.py
        useful_channel  = stream[ci]
        useful_channel.filter('lowpass', freq=8000)
        useful_channel = stream[ci].copy()

        trace_start = 30
        useful_channel.data = useful_channel.data[trace_start:]
        # TODO: see how performance affected chopping only when CASSM?
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

        if (plot_wiggles):
            plot_wiggle(pca_wiggle)
            #plot_wiggle(reconstructed_wiggle)

        multi_channel_pca_wiggle.extend(pca_wiggle)

    return multi_channel_pca_wiggle

# Given a 1D list of pca-ified wiggle(s), unleash provided model on data
def svm_predict(data_to_predict, model):
    data_predictions = model.predict_proba(data_to_predict)                 # magic function to return prediction probabilities of supplied model on data
    predictions      = []
    class_prediction = []

    for key, value in class_dict.items():
            class_prediction = (key, '%0.04f' % data_predictions[0][value])
            predictions.append(class_prediction)

    return sorted(predictions,key=lambda x: x[1], reverse=True)             # return class probabilities in descending order

# Given an .mseed timestamp and its class, check to see whether this .mseed 
# exists in a directory (likely one used for training) to classify on novel data
def check_mseed_existence(timestamp, mseed_dir):
    for root, dirs, mseeds in os.walk(mseed_dir):
        mseed_png_name = timestamp + "-L.mseed"
        if mseed_png_name in mseeds:
            return 1
    return 0

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

# Utility function to cleanly print out misclassified distribution of a class in ascending order
def print_misclassification_breakdown(total_class_misclassifications):
    for cl, class_breakdown in enumerate(total_class_misclassifications):
        class_label = find_class_label_from_value(cl, single_class=1)
        print("\n%s misclassification breakdown" % class_label, end="")
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
            print("%6s" % channel[1].ljust(6), end=" + ")
        print("       Total   Acceptability", end="\n\t  ")
        for uc in range(len(useful_channels)+1):
            print("-----------------", end="")
        print("")
        #print("".join(["--------------" for uc in range(useful_channels+1)]))
        for cp, class_prediction in enumerate(total_class_predictions):
            title_stream = "{:9s}".format(classes[cp].rjust(9).upper())
            print(title_stream, end="[")

            channel_stream = ""
            for p in channel_prediction:
                channel_stream += "%4d " % p
            else:
                print(channel_stream, end="")
            print("] E=%0*d" % (len(str(max_stream_count)), sum(channel_prediction)), end=" ")
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

def main():
    global expected_data_size
    global total_data
    global class_labels
        
    window_size      = 1300
    trigger_offset   = 300
    block_size       = 2
    n_cmpts          = floor(window_size / block_size) - 1

    # Grid search for PCA params
    window_sizes     = [900, 1100, 1200, 1300, 1500]
    block_sizes      = [2, 4, 8]
    trigger_offsets  = [100,200,300]

    do_train                = 1
    do_svm_grid_search      = 0
    iter_pca_params         = 1
    classify_single_mseed   = 0
    mass_mseed_classify     = 1
    
     model_name = "SVM-Classifier(%2d)_%3d_%1d_%2d_%1d" % \
    (max_stream_count, n_cmpts, block_size, window_size, trigger_offset)

    classes = ['meq', 'cassm', 'drilling','ert']
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
                                train_these_featuers(tf, do_svm_grid_search)

                                if mass_mseed_classify:
                                    mass_mseed_classification(0.72, 0.5, 1222, model_name, classes)

        else:
            expected_data_size = 0
            total_data   = []
            class_labels = []
            tf = TrainingFeatures(window_size, trigger_offset, block_size)
            train_these_featuers(tf, do_svm_grid_search)

    if classify_single_mseed:
        stream_path = "D:\\trigger\\1543378742.86.mseed"                # CASSM
        #stream_path = "D:\\trigger\\1544197621.54.mseed"               # Drilling
        #stream_path = "D:\\trigger\\1544136427.33.mseed"               # MEQ
        #stream_path = "D:\\trigger\\1543366796.60.mseed"               # ERT
        #stream_path = "D:\\trigger\\1544203603.19.mseed"
        #stream_path = "D:\\trigger\\1543365081.20.mseed"
        #classify_mseed(stream_path, model_name, num_channels=len(useful_channels))

    # TODO: - perform mass classification on all raw triggers, verify using all pngs    
    #       - manually separate training & validation mseeds to find true raw mass class. score

    # Iterates through each useful channel for every class and prints respective model score
    if (mass_mseed_classify):
        correct_threshold       = 0.72          # predictions above this are considered confidently correct
        mmmeh_threshold         = 0.50          # predictions below ^ and above this are ostensibly correct               
        max_classify_cnt        = 40            # max. amt. of .mseeds to classify per class (subject to avail.)

        mass_mseed_classification(correct_threshold, mmmeh_threshold, max_classify_cnt, model_name, classes)


if __name__ == '__main__':
    main()
