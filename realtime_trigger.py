import vibbox
from obspy import read
import logging, logging.handlers
import pandas as pd 
import os
import joblib
from SVM_v3 import svm_classify_stream
from tqdm import trange

logger = logging.getLogger()
#logger.basicConfig(filename='C:\\Users\\David\\Desktop\\logger.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# fname eg: D:\\Collab\\dat\\20180522\\vbox_201805222153378663.dat
model_path = "C:\\Users\\David\\Documents\\SVM\\nice_models\\"
model_name = "SVM-Classifier(3327)_849_2_1700_300"

probably_meq      = []
probably_ert      = []
probably_cassm    = []
probably_drilling = []
unsure            = []

confidence_threshold = 0.7

def process_file_trigger(fname):

    global probably_meq
    global probably_ert
    global probably_cassm
    global probably_drilling
    global unsure

    try:
        #st = vibbox.vibbox_read(fname)                                          # -- filtering
        st = read(fname)
        logger.info('Read ' + fname)
    except Exception as e:
        logger.info(e)
        logger.info('Cannot read ' + fname)
    try:
        pass
        st = vibbox.vibbox_preprocess(st)                                       # -- filtering
    except Exception as e:
        logger.debug(e)
        logger.debug('Error in preprocessing ' + fname)
    try:                                                                        
        if st[0].stats.npts < 10000:                                            # not .dat
            #classify_mseed(fname, model_name, do_validation=1)
            model = joblib.load(model_path+model_name)
            predictions = svm_classify_stream(st, model, model_name, class_type=-1)
            top_pred = predictions[0]
            #top_prediction = "%0.03f%% sure %s is %s" % (100*float(top_pred[1]), fname, top_pred[0])
            #print(top_prediction)
            if float(top_pred[1]) > confidence_threshold:   #confidence
                if top_pred[0] == 'MEQ':
                    probably_meq.append(fname)
                elif top_pred[0] == 'ERT':
                    probably_ert.append(fname)
                elif top_pred[0] == 'CASSM':
                    probably_cassm.append(fname)
                elif top_pred[0] == 'DRILLING':
                    probably_drilling.append(fname)
            else:
                unsure.append(fname)
                return
        
        #new_triggers = vibbox.vibbox_trigger(st.copy(), num=20)             # given Stream, finds location of triggers across all traces
        #new_triggers = vibbox.vibbox_checktriggers(new_triggers, st.copy())
        #new_triggers = svm_classify_dat(new_triggers, st.copy())
        
        #res = new_triggers.to_csv(header=False, float_format='%5.4f', index=False)
    except Exception as e:
        logger.debug(e)
        logger.debug('Error in triggering ' + fname)
        return 0
    try:
        pass
        st = vibbox.vibbox_custom_filter(st)
    except Exception as e:
        logger.debug(e)
        logger.debug('Error in filtering ' + fname)
    try:
        pass
        for index, ev in new_triggers.iterrows():
            ste = st.copy().trim(starttime = ev['time'] - 0.01,  endtime = ev['time'] + ev['duration'] + 0.01)
            outname = fname + '{:10.2f}'.format(ev['time'].timestamp)  + '.mseed'
            ste.write(outname, format='mseed')
    except Exception as e:
        logger.info(e)
        logger.info('Error in saving ' + fname)
        return 0
    return new_triggers

def main():
    fname = "D:\\Collab\\dat\\20181221\\vbox_201812212132324071.dat"
    fname = "D:\\trigger\\1543364527.08.mseed"
    mseed_directory = "D:\\trigger\\"
    max_mseeds        = 48000
    mseeds_considered = 0

    for root, dirs, mseeds in os.walk(mseed_directory):
        #random.shuffle(mseeds)
        if max_mseeds > len(mseeds):
            max_mseeds = len(mseeds)

        for mseed_index in trange(max_mseeds, leave=True):
                if mseed_index < max_mseeds:
                    mseed = mseeds[mseed_index]
                    df = process_file_trigger(root+mseed)
                else:
                    break                    

    # st = vibbox.vibbox_read(fname)
    # st[55:60].plot(method='full')
    
    # df = process_file_trigger(fname)    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
       print(df)

    print ("%d MEQs\n%d CASSM\n%d ERT\n%d Drilling \nfound in %d mseeds from %s" % (len(probably_meq), len(probably_cassm), len(probably_ert), len(probably_drilling), max_mseeds, root))
    print ("%d unsure classes (< %.02f similarity)" % (len(unsure), confidence_threshold))
    classes = ['meq', 'cassm', 'ert', 'drilling']
    for c in classes:
        file_name = "C:\\Users\\David\\Documents\\SVM\\raw_trigger_classifications\\probably_%s.txt" % c
        with open(file_name, 'a+') as the_file:
            if c == 'meq':
                for meq in probably_meq:
                    the_file.write(meq + "\n")
            elif c == 'cassm':
                for cassm in probably_cassm:
                    the_file.write(cassm + "\n")
            elif c == 'ert':
                for ert in probably_ert:
                    the_file.write(ert + "\n")
            elif c == 'drilling':
                for drilling in probably_drilling:
                    the_file.write(drilling + "\n")
    for hmm in unsure:
        file_name = "C:\\Users\\David\\Documents\\SVM\\raw_trigger_classifications\\unsure.txt"
        with open(file_name, 'a+') as the_file:
            the_file.write(hmm + "\n")


if __name__ == "__main__":
    main()