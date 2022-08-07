import pandas as pd
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def TSNE_post_process(Tsne_result, tonal_data, rhythm_data):
    logging.debug('Start TSNE_results_post_process')
    TSNE_DF = pd.DataFrame(Tsne_result)
    TSNE_DF.reset_index(drop = True, inplace = True)

    extra_info = tonal_data[['recording_artist','recording_name','genre1','kmeans_result']].copy()
    extra_info.reset_index(drop = True, inplace = True)

    rhythm_info = rhythm_data[['rhythm.bpm$float','rhythm.onset_rate$float','rhythm.beats_loudness.median$float','rhythm.danceability$float','rhythm.beats_count$int']]
    rhythm_info.reset_index(drop = True, inplace = True)

    extended_TSNE = pd.concat([TSNE_DF,extra_info,rhythm_info],axis = 1)

    # Set index
    TSNE_DF = TSNE_DF.set_index(tonal_data.index)

    # Set index
    extended_TSNE = extended_TSNE.set_index(tonal_data.index)

    TSNE_means_df = extended_TSNE.groupby(['recording_artist'], as_index=False).median()
    logging.debug('Finish TSNE_results_post_process')
    return(extended_TSNE, TSNE_DF, TSNE_means_df)

