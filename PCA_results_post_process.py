import pandas as pd
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def PCA_post_process(tonal_pca_mfcc, tonal_data, rhythm_data):
    logging.debug('Start PCA_results_post_process')
    #Create a df with DM results and recording info
    PCA_df = pd.DataFrame(tonal_pca_mfcc)
    extra_info = tonal_data[['recording_artist','recording_name','genre1','is_cloud','recording_album']].copy()
    extra_info.reset_index(drop = True, inplace = True)

    rhythm_info = rhythm_data[['rhythm.bpm$float','rhythm.onset_rate$float','rhythm.beats_loudness.median$float','rhythm.danceability$float','rhythm.beats_count$int']]
    rhythm_info.reset_index(drop = True, inplace = True)

    #Full df to work with
    extended_PCA_df = pd.concat([PCA_df,extra_info,rhythm_info],axis = 1)

    # Set index
    PCA_df = PCA_df.set_index(tonal_data.index)

    # Set index
    extended_PCA_df = extended_PCA_df.set_index(tonal_data.index)

    #Get mean values by artist
    PCA_means_df = extended_PCA_df.groupby(['recording_artist'],as_index=False).median().copy()

    logging.debug('Finish PCA_results_post_process')
    return(PCA_df, extended_PCA_df, PCA_means_df, rhythm_info)
