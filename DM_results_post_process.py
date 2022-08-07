import pandas as pd
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def DM_post_process(tonal_dm_mfcc, tonal_data, rhythm_data):
    logging.debug('Start DM_results_post_process')
    #Create a df with DM results and recording info
    DM_df = pd.DataFrame(tonal_dm_mfcc)
    extra_info = tonal_data[['recording_artist','recording_name','genre1','is_cloud','recording_album']].copy()
    extra_info.reset_index(drop = True, inplace = True)

    rhythm_info = rhythm_data[['rhythm.bpm$float','rhythm.onset_rate$float','rhythm.beats_loudness.median$float','rhythm.danceability$float','rhythm.beats_count$int']]
    rhythm_info.reset_index(drop = True, inplace = True)

    #Full df to work with
    extended_DM_df = pd.concat([DM_df,extra_info,rhythm_info],axis = 1)

    # Set index
    DM_df = DM_df.set_index(tonal_data.index)

    # Set index
    extended_DM_df = extended_DM_df.set_index(tonal_data.index)

    #Get mean values by artist
    DM_means_df = extended_DM_df.groupby(['recording_artist'],as_index=False).median().copy()

    logging.debug('Finish DM_results_post_process')
    return(DM_df, extended_DM_df, DM_means_df, rhythm_info)
