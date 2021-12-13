from utils import weight_my_delay, convert_time, aggr_delay
import pandas as pd
from math import isnan
# before creating our features we have to clearify one fact:
# since we kind of learn from history (see weighted delays from days before) we can obviously use the 
# "historic data" from train split BUT we cannot use it from our test split because
# in reality, we probably also have no access to that data (you dont know the future)! 
# One can argue: Well but you will always have historic data from older data sets. So not using the all
# historic data is also stupid! 

# Thats true, but we dont want to make our models too promising by basically learning AND evaluating on 
# the same data, because we also included the test data set for building our historic delay features
# if we would evauluate on a data set from a different year, say 2020, than we could learn the whole historic 
# data from 2015. here we are learning AND evaluating on 2015. Therefore we have to do it this way...
# this is also stated in issue #12

# some issues still holds:
# what happens when there is no historic data for our day interval of interest?
# - One simple solution: double the day interval until we have a value! (= fast convergence)
# what if one airport / airline comes up in the test data, that was not in our train data?
# - Well, then we basically know nothing. We should just give it the avg delay for now
# what if this airport / airline was in the train_set BUT we have no historic data for that day interval?
# - than we just take the feature from the closest day!

def delay_maker(train_set, test_set,days=1):
    # as described above, make the global avg from train
    train_global_avg = { "ORIGIN_AIRPORT": train_set["DEPARTURE_DELAY"].mean(),
                         "AIRLINE" : train_set["AIRLINE_DELAY"].mean(),
                       }

    # lets group the data by values of interest 
    groups = {
        "ORIGIN_AIRPORT" : train_set.groupby("ORIGIN_AIRPORT"),
        "AIRLINE" : train_set.groupby("AIRLINE")
        }
    
    # our function to do the heavy lifting of getting the avg delay of last X days
    # fast convergence as described above
    def _train_delay_maker(group_key ,value, current_day, delay_of_interest):
        ret = None
        true_days = days
        # find the avg delay_of_interest or double the interval
        while ret == None:
            ret = aggr_delay(groups, group_key , value, current_day, delay_of_interest, true_days)
            true_days += true_days
        return ret
    
    # helper function to get the next closest day 
    # if we can find our value but not any data for that column, we should use delays from closest day
    # not just take the avg
    def _find_neighbour_feature(df, colname , value):
        
        exactmatch = df[df[colname] == value]
        if not exactmatch.empty:
            return value
        else:
            lower_bound = df[df[colname] < value][colname].max()
            #print('l'+str(colname)+str(lower_bound)+' '+str(value))
            upper_bound = df[df[colname] > value][colname].min()
            #print('u'+str(colname)+str(upper_bound)+' '+str(value))
            dist_lower = abs(value - lower_bound) if not isnan(lower_bound) else 1000 # should be big enough 
            dist_higher = abs(value - upper_bound) if not isnan(upper_bound) else 1000 # should be big enough
            result = lower_bound if dist_lower <= dist_higher else upper_bound
            #print(result)
            return result
        
    # our function to build the delay feature for our test_set ONLY with knowlegde from the train split
    def _test_delay_maker(group_key, value, current_day, feature_of_interest):
        # if that key, value pair was not in train_split, use the global avg delay
        filtered_key = train_set.query("{} == @value".format(group_key))
        #print(filtered_key)
        if filtered_key.empty:
            return train_global_avg[group_key]
        # if it is there, we just take the feature for the closest day to current_day
        day_yearly = _find_neighbour_feature(filtered_key, "DAY_YEARLY", current_day)
        #print(day_yearly)
        feature = filtered_key.query("DAY_YEARLY == @day_yearly")[feature_of_interest].values[0]
        #print(feature)
        return feature
        
    # one helper function which wraps the whole building features into one function
    # will run on ORIGIN_AIRPORT, AIRLINE
    def _train_feature_maker_wrapper(value_dict, current_day):
        #def _train_departure_delay_maker(group_key ,value, current_day, delay_of_interest)
        w_dep_delay = _train_delay_maker("ORIGIN_AIRPORT", value_dict["ORIGIN_AIRPORT"], 
                                                   current_day, "DEPARTURE_DELAY")

        w_air_delay = _train_delay_maker("AIRLINE", value_dict["AIRLINE"], 
                                                   current_day, "AIRLINE_DELAY")
        result = pd.Series([w_dep_delay, w_air_delay]) # "DEPD", "AIRD"
        return result
    
    # one helper function which wraps the whole building 3 features into one function but for test_set
    # will run on ORIGIN_AIRPORT, DESTINATION_AIRPORT, AIRLINE
    def _test_feature_maker_wrapper(value_dict, current_day):
        #def _test_departure_delay_maker(group_key, value, current_day, feature_of_interest):
        #print("dep_delay")
        w_dep_delay = _test_delay_maker("ORIGIN_AIRPORT", value_dict["ORIGIN_AIRPORT"], 
                                                   current_day, "DEPD")
        #print("air_delay")
        w_air_delay = _test_delay_maker("AIRLINE", value_dict["AIRLINE"], 
                                                   current_day, "AIRD")
        result = pd.Series([w_dep_delay, w_air_delay]) # "DEPD", "AIRD"
        return result
    
    
    # first we apply the feature building process on the train_split
    # _train_departure_delay_maker(group_key ,value, current_day, delay_of_interest)
    print("\tcheck htop to spot deadlocks (should not happen)")
    print("\tApplying feature extraction to train_set on DEPARTURE_DELAY, AIRLINE_DELAY...")
    train_set[["DEPD", "AIRD"]] = train_set.parallel_apply(lambda i:
                                               _train_feature_maker_wrapper(
                                                        i[["ORIGIN_AIRPORT", "AIRLINE"]],
                                                        i["DAY_YEARLY"]), 
                                                axis=1)
    # now we use the train_split to augment our test_split
    
    #print(train_set)
    print("\tApply feature extraction to test_set from train_set...")
    test_set[["DEPD", "AIRD"]] = test_set.parallel_apply(lambda i:
                                                _test_feature_maker_wrapper( 
                                                        i[["ORIGIN_AIRPORT", "AIRLINE"]],
                                                        i["DAY_YEARLY"]), 
                                                axis=1)
    #print(split[1])
    return train_set, test_set

