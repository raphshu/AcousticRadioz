import pandas as pd
import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def SCL_ADM_post_process(tonal_scalable_adm_mfcc, tonal_data, rhythm_data):
    logging.debug('Start SCL_ADM_results_post_process')

    #Create a df with ADM results and recording info
    Tonal_scalable_ADM_df = pd.DataFrame(tonal_scalable_adm_mfcc)

    #Add metadata
    extra_info = tonal_data[['recording_artist','recording_name','genre1','is_cloud','recording_album','kmeans_result']].copy()
    extra_info.reset_index(drop = True, inplace = True)
    rhythm_info = rhythm_data[['rhythm.bpm$float','rhythm.onset_rate$float','rhythm.beats_loudness.median$float','rhythm.danceability$float','rhythm.beats_count$int']]
    rhythm_info.reset_index(drop = True, inplace = True)

    #Add rhythm data to the tonal adm result
    extended_Tonal_scalable_ADM_df = pd.concat([Tonal_scalable_ADM_df,extra_info,rhythm_info],axis = 1)

    # Set index
    Tonal_scalable_ADM_df = Tonal_scalable_ADM_df.set_index(tonal_data.index)

    # Set index
    extended_Tonal_scalable_ADM_df = extended_Tonal_scalable_ADM_df.set_index(tonal_data.index)

    #Calculate the media vectors of ADM - per artist
    Tonal_scalable_ADM_means_df = extended_Tonal_scalable_ADM_df.groupby(['recording_artist'],as_index=False).median()

    # Count number of songs for each artist - by genre
    genres = tonal_data.groupby(['recording_artist', 'genre1', 'genre2'], as_index=False).count()

    # Join ADM result and genres
    artist_genre_df = pd.merge(Tonal_scalable_ADM_means_df, genres, left_on='recording_artist', right_on='recording_artist')

    # Sort values by number of shows for each genre
    artist_genre_df = artist_genre_df.sort_values(["recording_artist", "recording_id"], ascending=(False, False))

    # drop duplicate genres - Leave only the most popular genre for each artist
    artist_genre_df = artist_genre_df.drop_duplicates(subset='recording_artist')

    # drop columns - Leave only relevant ones
    artist_genre_df = artist_genre_df[
        ['recording_artist', 0, 1, 2, 'genre1', 'genre2', 'recording_album', 'rhythm.bpm$float',
         'rhythm.onset_rate$float', 'rhythm.beats_loudness.median$float', 'rhythm.danceability$float',
         'rhythm.beats_count$int']]

    logging.debug('Finish SCL_ADM_results_post_process')
    return (tonal_data, rhythm_data, Tonal_scalable_ADM_df, Tonal_scalable_ADM_means_df,
            extended_Tonal_scalable_ADM_df, artist_genre_df)