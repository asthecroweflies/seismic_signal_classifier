import os
import shutil

from math import floor
import random
from tqdm import trange

'''
    Given a directory full of .mseed files, randomly split the data into
    two directories at a specified proportion tuple.

'''
# e.g. params
#    D:\labeled_data\optimized_and_labeled_triggers\agnostic_triggers\sequential\meq
#    D:\labeled_data\reserved\meq_training
#    D:\labeled_data\reserved\meq_testing
#    (70,30)
def split_mseed_data(host_dir, training_dir, testing_dir, train_test_split):
    training_data = []
    testing_data  = []
    mseed_count = len([f for f in os.listdir(host_dir) if os.path.isfile(os.path.join(host_dir, f))])
    max_training_size = floor(train_test_split[0] * mseed_count)

    for root, dirs, mseeds in os.walk(host_dir):
        random.shuffle(mseeds)
        for m in trange(len(mseeds), leave=True):
            mseed = mseeds[m]
            if (len(training_data) < max_training_size) and (mseed not in training_data):
                training_data.append(mseed)
                shutil.copy(host_dir + '//' + mseed, training_dir)
            elif (mseed not in testing_data):
                testing_data.append(mseed)
                shutil.copy(host_dir + '//' + mseed, testing_dir)

def main():
    classes = ['cassm', 'ert', 'meq', 'drilling']
    for c in classes:
        host_dir     = "D:\\labeled_data\\optimized_and_labeled_triggers\\agnostic_triggers\\%s" % c
        training_dir = "D:\\labeled_data\\optimized_and_labeled_triggers\\training_mseeds\\%s_training\\" % c
        testing_dir  = "D:\\labeled_data\\optimized_and_labeled_triggers\\validation_mseeds\\%s_testing\\" %c
        if (os.path.isdir(training_dir) == 0) and (os.path.isdir(testing_dir) == 0):                    # make directory if nonexistent
            try:
                os.makedirs(training_dir)
                os.makedirs(testing_dir)
            except Exception as e:
                print("could not make directory [%s]" % (e))
        split_mseed_data(host_dir, training_dir, testing_dir, (0.65, 0.35))

if __name__ == "__main__":
    main()