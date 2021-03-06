
import pandas as pd
import numpy as np
import os
import pickle
from scripts.utils import read_list_from_text

def extract_hour_from_time_string(df:pd.core.frame.DataFrame,
                                  hour_column:str, 
                                  sep:str='_',
                                  format_string:str = '%H:%M:%S')->list:
    """
    Extracts hour from the passed hour column separated by the given character.
    
    """
    try:
        if sep=='_':
            return pd.to_datetime(df[hour_column].apply(lambda x: str(x).split('-')[0].strip()), format=format_string).dt.hour
        else:
            raise NotImplementedError      
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))


def combine_two_ids_to_one_unique_id(df:pd.core.frame.DataFrame,
                                      column_1:str='campaign_id',
                                      column_2:str='ad_id',
                                      prefix1:str='campaign_id',
                                      prefix2:str='ad_id')->list:
    """
    Combines the two given column and returns by adding prefixes-> prefix1+column1+'_'+prefix2+column2
    
    """
    try:
        df[column_1] = [prefix1+'_' + str(i) for i in df[column_1]]
        df[column_2] = [prefix2 + str(i) for i in df[column_2]]
        return df[[column_1, column_2]].apply(lambda x: '_'.join(x), axis=1)
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))

def extract_start_end_hour_per_id(df:pd.core.frame.DataFrame,
                                   id_column:str='unique_ids',
                                   datetime_column:str='datetime',
                                   start_hour_column_name:str = 'first_hour',
                                   end_hour_column_name:str = 'endtime',   
                                  )->pd.core.frame.DataFrame:
    """
    Given the appropiate column names this function groups the 
    rows by id and extracts start and endtime of the id in the 
    dataframe. 

    """   
    try:
        unique_campaign_advertisement_info = df[[id_column, datetime_column]].sort_values([datetime_column]).groupby(id_column).first().reset_index()
        unique_campaign_advertisement_info = unique_campaign_advertisement_info.rename(columns={datetime_column:start_hour_column_name})

        #extract end hour of of each series
        unique_campaign_advertisement_info = unique_campaign_advertisement_info.set_index(id_column)
        unique_campaign_advertisement_info[end_hour_column_name] = df.sort_values(datetime_column).groupby(id_column).last()[[datetime_column]]

        return unique_campaign_advertisement_info
    
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))

def get_ads_lasting_n_hours(df:pd.core.frame.DataFrame,
                            start_time:str,
                            end_time:str,
                            n:int=25)->list:
    """
    Returns the ids of advertisements lasting atleast n hours. 
    
    """   
    try:
        df['duration_hours']=(df[end_time]-df[start_time]).astype('timedelta64[h]')
        df[df['duration_hours']>=n]
        return df[df['duration_hours']>=n].index.values
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))


def interpolate_missing_hours(series:pd.core.frame.DataFrame, 
                              date_column:str, 
                              group_name:str,
                              fill_value:int = 0)->pd.core.frame.DataFrame:
    """
    Returns an updated dataframe filling the missing hours by fill_value.
    
    """ 
    try:
        series=series.set_index(date_column) 
        series = series.resample('H').asfreq()

        series['unique_ids'] = group_name
        series = series.fillna(fill_value).reset_index()

        return series
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))
        
def set_train_test_identifier(series:pd.core.frame.DataFrame,
                              test_set_end:pd._libs.tslibs.timestamps.Timestamp,
                              date_column:str,
                              last_hour:int,
                              train_set_identifier:str,
                              test_set_identifier:str)->pd.core.frame.DataFrame:
    """
    Returns an updated dataframe to identify rows associated with train and test
    dataset by boolean value. training rows must be <=last_hour and test rows are
    opposite.
    
    """ 
    try:
        #set boolean columns to identify train anad test set
        series[train_set_identifier] = series[date_column]<=last_hour
        series[test_set_identifier] =(series[date_column]>last_hour) & (series[date_column]<=test_set_end)
        return series
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))
        
def extract_and_interpolate_valid_ads(df:pd.core.frame.DataFrame,
                                      unique_ads:pd.core.frame.DataFrame,
                                      date_column:str='datetime',
                                      train_set_identifier:str='first_24_hour',
                                      test_set_identifier:str='test_set',
                                      unique_id_name:str='unique_ids',
                                      test_days_per_ad:int=3)->pd.core.frame.DataFrame:
  
    """
    Groups each id and sorts by date_column which has 24+test_days_per_ad hours of info
    and interpolate in between missing values. Furthermore identifies rows belonging to 
    train or test group and returns the updated dataframe.
    
    Args:
        test_days_per_ad:int -> days beyond 24 hours
    """

    index = 1
    prepared_df = None
    hours_found_in_dataset = []
    try:
        grouped_unique_ids = df.sort_values(['datetime']).groupby('unique_ids')

        for group_name,series in grouped_unique_ids:
            if group_name in unique_ads.index.values:

                hours_found_in_dataset.append(series.shape[0])
                print('{}. Valid Ids {}: duration in hours {}'.format(index, group_name, series.shape[0]))  

                last_hour = unique_ads[unique_ads.index == group_name]\
                            .last_hour.values[0]

                test_set_end = last_hour+pd.to_timedelta(test_days_per_ad, unit='h')

                series = interpolate_missing_hours(series.copy(), date_column, group_name)
                series = set_train_test_identifier(series,
                                                   test_set_end,
                                                   date_column,
                                                   last_hour,
                                                   train_set_identifier,
                                                   test_set_identifier)

                prepared_df = series if index==1 else prepared_df.append(series)
                index+=1

        return prepared_df,hours_found_in_dataset
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))


def get_ctr(prepared_df:pd.core.frame.DataFrame,
                  click_column:str='clicks',
                  impression_column:str='impressions')->list:
    """
    Calculates clickthrough rate i.e. clicks/impressions.
    
    """ 
    try:
        temp_df = pd.DataFrame({'ctr' : []})
        temp_df['ctr'] = prepared_df[click_column]/prepared_df[impression_column]*100
        temp_df.loc[temp_df['ctr'] ==np.inf, 'ctr'] = 100
        temp_df.loc[temp_df['ctr']>100, 'ctr'] = 100
        temp_df['ctr'] = temp_df['ctr'].fillna(0)
        return temp_df['ctr']
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))

def get_conversion_rate(prepared_df:pd.core.frame.DataFrame,
                              click_column:str='clicks',
                              purchase_column:str='purchase')->list:
    """
    Calculates conversion rate i.e. purchase/clicks.
    
    """

    try:
        temp_df = pd.DataFrame({'conversion_rate' : []})
        temp_df['conversion_rate'] = prepared_df[purchase_column]/prepared_df[click_column]*100

        temp_df['conversion_rate'] = prepared_df['purchase']/prepared_df['clicks']*100
        temp_df.loc[temp_df['conversion_rate'] ==np.inf, 'conversion_rate'] = 100
        temp_df.loc[temp_df['conversion_rate']>100, 'conversion_rate'] = 100
        temp_df['conversion_rate'] = temp_df['conversion_rate'].fillna(0)

        return temp_df['conversion_rate']
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))
        
def get_custom_conversion_rate(prepared_df:pd.core.frame.DataFrame,
                                     ctr_column:str='ctr',
                                     conversion_rate_column:str='conversion_rate',
                                     alpha_value:int = 2)->list:
    """
    Calculates custom conversion rate i.e. (click through rate +conversion rate)/alpha+1
    
    """
    try:
        return (prepared_df[ctr_column]+(prepared_df[conversion_rate_column]*alpha_value))/(alpha_value+1)
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))


def get_prepared_data_summary(dataframe:pd.core.frame.DataFrame,
                    training_set_column_indicator:str='first_24_hour',
                    test_set_column_indicator:str='test_set',
                    target_column:str= 'next_hour_good_performance')->None:
    """
    Prints info regarding train/test and good/bad performing advertisements in 
    train and test datasets. 
    
    """
    try:
        training_set = dataframe[dataframe[training_set_column_indicator]]

        print('Training data %: ', (100*training_set.shape[0])/dataframe.shape[0])
        
        good_ads = training_set[training_set[target_column]>0].shape[0]

        print('Training data ads performing good: ', good_ads)
        print('{}% w.r.t training'.format((100*good_ads)/training_set.shape[0]))

        del training_set

        test_set = dataframe[dataframe[test_set_column_indicator]]

        print('\nTest data total ads: ', test_set.shape[0])
        print('Number of ads with good performance: {}'.format(test_set[test_set[target_column]>0].shape[0]))
        print('{}%'.format((100*test_set[test_set.next_hour_good_performance>0].shape[0])/test_set.shape[0]))
        
        del test_set
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))

def get_feature_list_from_text_excluding_target(feature_list_path:str,
                                                target_column:str)->list:
    """
    Reads list fro the given text file. Removes target column from it and
    returns the list

    """  

    try:
        feature_list = read_list_from_text(feature_list_path)
        feature_list.sort(reverse=True)
        
        if target_column in feature_list:
            feature_list.remove(target_column)
            
        return feature_list
    
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))