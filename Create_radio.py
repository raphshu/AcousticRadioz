import numpy as np
import pandas as pd
import math
import random

from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from scipy import stats
from sklearn.neighbors import NearestNeighbors

from TEMPO_measure import TEMPO_measure
import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def create_radio(Tempo_compare,diversity, number_of_songs_in_radio,Algorithm, number_of_tonal_clusters,
                tonal_data, bpm_onset, cloud_artist, Tonal_ADM_df, Tonal_ADM_means_df,
                DM_df, DM_means_df,
                TSNE_DF, TSNE_means_df, Tonal_scalable_ADM_df, Tonal_scalable_ADM_means_df,
                PCA_df, PCA_means_df):
    logging.debug('Start Creating RADIO')

    number_of_artists_in_radio = (2**diversity + (diversity-1))
    percentage_of_songs_for_each_artist = ((100/number_of_artists_in_radio)/100) * 2
    # percentage_of_songs_for_each_artist = 100
    number_of_songs_for_each_seed_song = math.ceil(number_of_songs_in_radio/number_of_tonal_clusters)+1


    # print('Number of similar songs to each seed song: '+str(number_of_songs_for_each_seed_song))

    max_songs_per_artist = percentage_of_songs_for_each_artist * number_of_songs_in_radio
    # print('max songs per artist: ' + str(max_songs_per_artist))
    # print('')

    if Algorithm == 'ADM':
        means_df = Tonal_ADM_means_df.iloc[:,:4]
        reduction_result_df = Tonal_ADM_df
    elif Algorithm == 'PCA':
        means_df = PCA_means_df.iloc[:, :4]
        reduction_result_df = PCA_df
    elif Algorithm == 'SCL-ADM':
        means_df = Tonal_scalable_ADM_means_df.iloc[:, :4]
        reduction_result_df = Tonal_scalable_ADM_df
    elif Algorithm == 'DM':
        means_df = DM_means_df.iloc[:,:4]
        reduction_result_df = DM_df
    else:
        means_df = TSNE_means_df.iloc[:,:4]
        reduction_result_df = TSNE_DF

    #KNN by artist
    seed_artist_nbrs = NearestNeighbors(n_neighbors = number_of_artists_in_radio, algorithm = 'ball_tree').fit(means_df.iloc[:,1:])
    distances, indices = seed_artist_nbrs.kneighbors(means_df[means_df['recording_artist'] == cloud_artist].iloc[:,1:])

    #Get similar artists
    tonal_similar_artists_index_in_means_df = indices[:,:].tolist()[0]
    tonal_similar_artists = means_df['recording_artist'].iloc[tonal_similar_artists_index_in_means_df].values.tolist()

    # print('The {} most similar artists to {} :'.format(number_of_artists_in_radio, cloud_artist))
    # print("")
    # for i in range(1, len(tonal_similar_artists)):
    #     print(str(i) + str('. ' + str(tonal_similar_artists[i])))

    ##### Filter only to similar artists data #####

    # Get similar songs index
    similar_artists_songs_index = tonal_data[tonal_data['recording_artist'].isin(tonal_similar_artists)].index

    # Keep only similar songs embedding
    similar_artists_3d = reduction_result_df.iloc[similar_artists_songs_index].copy()

##################################################################### I CHANGED 65 AND 69 FROM LOC TO ILOC !!!!!!!!!!!!!
    # Slice tonal data of similar songs
    similar_artists_tonal_data = tonal_data.iloc[similar_artists_songs_index].copy()

    #### KNN of seed songs ####
    # Activate KNN on the embedding result
    current_song_nbrs = NearestNeighbors(n_neighbors=31, algorithm='ball_tree').fit(similar_artists_3d)
    distances, indices = current_song_nbrs.kneighbors(similar_artists_3d)
    indices = pd.DataFrame(indices)
    indices['orig_index'] = similar_artists_songs_index

    similar_artists_tonal_data['orig_index'] = similar_artists_tonal_data.index
    # Create a DF with the results of one song from each tonal cluster
    # Initiate a list for knn results
    radio_nn = []

    # DF of the songs that were actually chosen to be a seed of the radio
    query_songs = pd.DataFrame()
    # For each tonal cluster - pick a song
    for j in range(number_of_tonal_clusters):
        # Extract the songs of the current tonal cluster
        curr_tonal = similar_artists_tonal_data.loc[similar_artists_tonal_data['kmeans_result'] == j]

        # Randomly sample a song from the current cluster
        current_cluster_sample = random.choice(curr_tonal.index.to_list())

        # Save the song to df
        query_songs[j] = similar_artists_tonal_data.loc[current_cluster_sample]

        # Extract the sample song Nearest Neighbors
        current_sample_nn = indices[indices['orig_index'] == current_cluster_sample].iloc[:, 1:-1].values[0]

        # Add the results to the list of NN
        radio_nn.append(current_sample_nn)

    # Flatten the list of lists
    radio_nn = [item for sublist in radio_nn for item in sublist]

    # Convert the list of seed songs Nearst Neighbors to DF
    nn_df = pd.DataFrame(radio_nn)

    # print('Seed songs by artist: ')
    # print('')
    # print(query_songs.transpose()['recording_artist'].value_counts())

    #### Create index list of most tonaly similar songs ####

    # Get running index of most tonaly similar songs
    most_tonaly_similar_songs_index = [item for sublist in nn_df.value_counts()[:600].index.tolist() for item in
                                       sublist]
    # print('most tonaly similar songs running index: ' + str(most_tonaly_similar_songs_index[:10]))
    # Convert to original index of most tonaly similar songs
    most_tonaly_similar_songs_index = indices['orig_index'].iloc[most_tonaly_similar_songs_index].values.tolist()
    # print('most tonaly similar songs original index: ' + str(most_tonaly_similar_songs_index[:10]))
    # get index of seed songs
    seed_songs_index = query_songs.transpose()['orig_index'].tolist()
    # print('seed songs index: ' + str(seed_songs_index))
    if not Tempo_compare:
        radio_result = tonal_data[['recording_artist', 'recording_name', 'genre1']].loc[
            most_tonaly_similar_songs_index]
        #     print(radio_result['recording_artist'].value_counts())
        # print('')
        # Filter result to match the artists diversity standart
        songs = 0
        final_radio_index = 0
        artists = {}
        final_radio = pd.DataFrame(columns=['recording_artist', 'recording_name', 'genre1'])



        while songs < number_of_songs_in_radio:
            curr_song = radio_result.iloc[final_radio_index, :]
            curr_artist = curr_song['recording_artist']

            # Check if the artist has exceeded the limit per artist - diversity
            if curr_artist in artists.keys():
                if artists[curr_artist] < (max_songs_per_artist - 1):
                    artists[curr_artist] += 1
                    songs += 1
                    final_radio = final_radio.append(curr_song)
                else:
                    final_radio_index += 1
                    continue
            else:
                artists[curr_artist] = 1
                final_radio = final_radio.append(curr_song)
                songs += 1
            final_radio_index += 1

    if Tempo_compare:
        # Merge seed songs index with most tonaly similar songs
        full_index_for_TEMPO_kernel =  most_tonaly_similar_songs_index + seed_songs_index
        # print('full_index_for_TEMPO_kernel index: ' + str(len(full_index_for_TEMPO_kernel)))
        # Slice BPM and Onset rate df to relevant songs
        Songs_for_TEMPO = bpm_onset.loc[full_index_for_TEMPO_kernel]

        # Computing a square matrix of TEMPO dist.
        tempo_mtx = squareform(pdist(Songs_for_TEMPO, TEMPO_measure))
        TEMPO_df = pd.DataFrame(tempo_mtx)

        # TEMPO outlier removal

        z_scores = stats.zscore(TEMPO_df)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        filtered_Tempo_df = TEMPO_df[filtered_entries]
        # print('TEMPO_df :'+str(TEMPO_df))
        # Keep only rows of seed songs - row 1 column 3 means how similar is song 3 to song 1 - TEMPO wise.
        TEMPO_df = TEMPO_df.iloc[:number_of_tonal_clusters, :]

        #Select dictionaries to control number of songs for each artist and to avoid song duplication
        artists = {}
        songs = {}
        current_number_of_songs = 0
        final_radio = pd.DataFrame(columns=['recording_artist', 'recording_name', 'genre1'])

        # Iterate on songs - include seed songs as posible result
        for song in TEMPO_df.index:
            songs_for_curr_seed_song = 0
            curr_seed_song = TEMPO_df.iloc[song]
            curr_seed_song = curr_seed_song.sort_values(ascending=True)

            for running_index in curr_seed_song.index:
                curr_song = tonal_data[['recording_artist', 'recording_name', 'genre1']].loc[
                    Songs_for_TEMPO.iloc[running_index].name]
                curr_artist = curr_song['recording_artist']
                # Check if the artist has exceeded the limit per artist - diversity
                if curr_artist in artists.keys():
                    # Check if that we didn't exceed the amount of songs for each artist
                    if (artists[curr_artist] <= (max_songs_per_artist - 1)) and (str(curr_song) not in songs.keys()):
                        artists[curr_artist] += 1
                        songs[str(curr_song)] = 1
                        current_number_of_songs += 1
                        final_radio = final_radio.append(curr_song)
                        songs_for_curr_seed_song += 1
                    else:
                        continue
                else:
                    artists[curr_artist] = 1
                    songs[str(curr_song)] = 1
                    final_radio = final_radio.append(curr_song)
                    current_number_of_songs += 1
                    songs_for_curr_seed_song += 1

                if songs_for_curr_seed_song == number_of_songs_for_each_seed_song:
                    break

                if current_number_of_songs == number_of_songs_in_radio:
                    break

            if current_number_of_songs >= number_of_songs_in_radio:
                break
    logging.debug('Finish Creating RADIO')
    return final_radio, final_radio.index, seed_songs_index