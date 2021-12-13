# our weighting function to punish delays that are further back in time
# linear decrease
# lets say days=3, than 
# => same day: full delay
# => one days: 0.66 delay
# => two days: 0.33 delay
# using this simple approach we simulate that airports try to fixed their issues or situations generally changes over time

def weight_my_delay(current_delay, current_day, day_max, days=1):
    multi = 1 / (days)
    diff_days = (day_max - current_day) % 365
    factor = 1 - (multi * diff_days)
    return current_delay * factor



# we have SCHEDULED_DEPARTURE and SCHEDULED_ARRIVAL in a very weird time format. 
# convert it to min since our delays are also in min
def convert_time(time): # legacy function... NOT IN USE, maybe remove it!
        time=int(time)
        e = str(time)
        #print(e)
        min = int(e[-2:])
        hours = int(e[:-2]) if e[:-2] != '' else 0
        return hours*60 + min

# we also need a function to aggregate delays for specific groups such as airports or airlines
# to do so we define this function that takes
# groups:               grouping dict with information from pandas.groupby
# group_key:            group key such as 'ORIGIN_AIRPORT' => groupby 'ORIGIN_AIRPORT'
# value:                actual value such as airline or airport
# current_day:          actual day of a specific data entry we want to learn
# delay_of_interest:    the delay we want to learn such as 'DEPARTURE_DELAY' 
# days(default=1):      the day interval to look back and learn from
# it will weight the delays using weight_my_delay and than return the custom weighted mean

import pandas as pd
def aggr_delay(groups, group_key,value, current_day, delay_of_interest,days=1):
    day_interval_low = (current_day - days) % 365
    sus = pd.DataFrame()
    if day_interval_low > current_day: # case when you have current_day=x and days>x and you "go back one year"
        sus = groups[group_key].get_group(value)[["DAY_YEARLY", delay_of_interest]].query(
                                        "DAY_YEARLY > @day_interval_low | DAY_YEARLY <= @current_day")
                                        
        if sus.empty:
            return None

        sus[delay_of_interest] = sus.apply(lambda i: weight_my_delay(i[delay_of_interest], 
                                                                     i["DAY_YEARLY"], current_day, days), axis=1)

    else:
        sus = groups[group_key].get_group(value)[["DAY_YEARLY", delay_of_interest]].query(
                                        "DAY_YEARLY > @day_interval_low & DAY_YEARLY <= @current_day")
                                        
        
        if sus.empty:
            return None
        
        sus[delay_of_interest] = sus.apply(lambda i: weight_my_delay(i[delay_of_interest], 
                                                                     i["DAY_YEARLY"], current_day, days), axis=1 )

    return sus[delay_of_interest].mean()