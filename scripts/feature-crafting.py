#!/usr/bin/env python

import sys, argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from hand_crafted import delay_maker
from mlp_crafted import estimator_maker
from utils import convert_time
from pandarallel import pandarallel

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn




# we define some cmdline args to control this script
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="Input .csv containing the data set")
parser.add_argument("-train", type=str, help="Output train set .csv to write the features into or 'console' to console")
parser.add_argument("-test", type=str, help="Output test set .csv to write the features into or 'console' to console")
parser.add_argument("-d", "--day", nargs='?', default=None, const=1, type=int, help="Number of days to look back")
parser.add_argument("--mean", action="store_true", help="mean mode: using manual data statistics to learn features")
parser.add_argument("--mlp", action="store_true", help="mlp mode: use MLPRegressor to extract features")


def main():
    seed = 433484 # hard coded seed
    pandarallel.initialize(progress_bar=False, verbose=0)
    input_file = pd.DataFrame()
    transformed_file = pd.DataFrame()
    train_file =pd.DataFrame()
    test_file = pd.DataFrame()
    args = parser.parse_args()
    from warnings import simplefilter
    from sklearn.exceptions import ConvergenceWarning
    simplefilter("ignore", category=ConvergenceWarning)


    pd.set_option("display.max_rows", None, "display.max_columns", None)
    if args.input:
        #reading input csv
        try:
            input_file = pd.read_csv(args.input)
            print("Loaded {}".format(args.input))
            transformed_file = input_file
            #print(transformed_file.head())
        except:
            print("No input file found...exit")
    if args.mean:
        # we extract the features by hand. see issue #12
        # first we take out the stuff we are interested in
        local_trans = transformed_file[["DAY_YEARLY", "DAY_OF_WEEK", "MONTH", "DISTANCE", "SCHEDULED_DEPARTURE", 
        "SCHEDULED_ARRIVAL", "SCHEDULED_TIME", "ORIGIN_AIRPORT", "AIRLINE",
        "DEPARTURE_DELAY", "AIRLINE_DELAY", "TARGET", "ARRIVAL_DELAY"]]
        if args.day:
            print("Will extract features using {} day interval, seed: {}".format(args.day, seed))
            print("For further information check issue #12")
            train_set, test_set = train_test_split(local_trans, test_size = 0.85, random_state=seed)
            train_file, test_file = delay_maker(train_set, test_set, days=args.day)

    if args.mlp:
        local_trans = transformed_file[["DAY_YEARLY", "DAY_OF_WEEK", "MONTH", "DISTANCE", "SCHEDULED_DEPARTURE", 
        "SCHEDULED_ARRIVAL", "SCHEDULED_TIME", "ORIGIN_AIRPORT", "AIRLINE",
        "DEPARTURE_DELAY", "AIRLINE_DELAY", "TARGET", "ARRIVAL_DELAY"]]

        print("Preparing: Converting time formats")
        local_trans["SD_MIN"] = local_trans.parallel_apply(lambda d: convert_time(d["SCHEDULED_DEPARTURE"]), axis=1)
        local_trans["SA_MIN"] = local_trans.parallel_apply(lambda d: convert_time(d["SCHEDULED_ARRIVAL"]), axis=1)

        print("Will extract features using MLPRegressor (24,16,8) seed: {}".format(seed))
        print("For further information check issue #12")
        train_set, test_set = train_test_split(local_trans, test_size=0.85, random_state=seed)
       
        train_file, test_file = estimator_maker(train_set, test_set)
     
    if args.train:
        # save the features
        if args.train == 'console':
            print("------------------\nTrain {}".format(len(train_file)))
            print(train_file.head())
        else:
            try:
                train_file.to_csv(args.train)
                print("Written train features to {} successfully".format(args.train))
            except:
                print("Could not write train features to {}".format(args.train))
    
    if args.test:
        
        # save the features
        if args.test == 'console':
            print("------------------\nTest {}".format(len(test_file)))
            print(test_file.head())
        else:
            try:
                test_file.to_csv(args.test)
                print("Written test features to {} successfully".format(args.test))
            except:
                print("Could not write test features to {}".format(args.test))
    
    print("Done")
    

     
if __name__ == "__main__":
    main()
