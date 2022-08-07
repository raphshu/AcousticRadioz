import logging
import pandas as pd
import pickle
import plotly.express as px

from Create_radio import create_radio
from Evaluation_CG import evaluation_cg

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.debug,
                    datefmt='%Y-%m-%d %H:%M:%S')

logging.debug('Start import data')
# coldplay , the beatles , jamiroquai , ludwig van beethoven
cloud_artist = 'coldplay'

# Import data for adm
tonal_data = pd.read_csv(
    r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_tonal_data.csv".format(cloud_artist))
bpm_onset = pd.read_csv(
    r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_bpm_onset.csv".format(cloud_artist))
adm_features = pd.read_csv(r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\adm_features.csv")

# number_of_tonal_clusters = (adm_features.columns[0]).strip('[')
# number_of_tonal_clusters = int(number_of_tonal_clusters)
number_of_tonal_clusters = 15

# No use of normal ADM
# Tonal_ADM_df = pd.read_csv(r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_Tonal_ADM_df.csv".format(cloud_artist),index_col = 'Unnamed: 0')
# Tonal_ADM_means_df = pd.read_csv(r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_Tonal_ADM_means_df.csv".format(cloud_artist),index_col = 'Unnamed: 0')
Tonal_ADM_df = None
Tonal_ADM_means_df = None

# Import Embedded results
Tonal_scalable_ADM_df = pd.read_csv(
    r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_Tonal_scalable_ADM_df.csv".format(
        cloud_artist), index_col='Unnamed: 0')
Tonal_scalable_ADM_means_df = pd.read_csv(
    r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_Tonal_scalable_ADM_means_df.csv".format(
        cloud_artist), index_col='Unnamed: 0')
extended_scalable_ADM_df = pd.read_csv(
    r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_extended_scalable_ADM_df.csv".format(
        cloud_artist), index_col='Unnamed: 0')
DM_df = pd.read_csv(
    r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_DM_df.csv".format(cloud_artist),
    index_col='Unnamed: 0')
DM_means_df = pd.read_csv(
    r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_DM_means_df.csv".format(cloud_artist),
    index_col='Unnamed: 0')
extended_DM_df = pd.read_csv(
    r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_extended_DM_df.csv".format(cloud_artist),
    index_col='Unnamed: 0')
TSNE_df = pd.read_csv(
    r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_TSNE_df.csv".format(cloud_artist),
    index_col='Unnamed: 0')
TSNE_means_df = pd.read_csv(
    r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_TSNE_means_df.csv".format(cloud_artist),
    index_col='Unnamed: 0')
extended_TSNE_df = pd.read_csv(
    r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_extended_TSNE_df.csv".format(
        cloud_artist), index_col='Unnamed: 0')
PCA_df = pd.read_csv(
    r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_PCA_df.csv".format(cloud_artist),
    index_col='Unnamed: 0')
PCA_means_df = pd.read_csv(
    r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_PCA_means_df.csv".format(cloud_artist),
    index_col='Unnamed: 0')
extended_PCA_df = pd.read_csv(
    r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_extended_PCA_df.csv".format(cloud_artist),
    index_col='Unnamed: 0')

logging.debug('Finish import data')

#Import the related artists to the seed artist - based on Melon playlist dataset
related_artists_df = pd.read_csv(
    r"C:\Users\raphs\Documents\University\msc\thesis\data\related_artists_from_melon\{}_related_artists_df.csv".format(
        cloud_artist))

# ------ Radio features ------#
diversity = 3
number_of_songs_in_radio = 50

# Set all Radio options
logging.debug('Start Creating Radios')

TEMPO = [True, False]
Algorithm_lst = ['SCL-ADM', 'PCA', 'DM', 'TSNE']
radios_count = 1
num_of_iterations_for_each_setting = 10
Radio_results = {}
Genres_dist = {}
Artists_dist = {}

for alg in Algorithm_lst:
    for T in TEMPO:
        radio_flag = True
        for iteration in range(num_of_iterations_for_each_setting):
            radio, radio_index, seed_index = create_radio(T, diversity, number_of_songs_in_radio,
                                                          alg, number_of_tonal_clusters,
                                                          tonal_data, bpm_onset, cloud_artist, Tonal_ADM_df,
                                                          Tonal_ADM_means_df,
                                                          DM_df, DM_means_df,
                                                          TSNE_df, TSNE_means_df, Tonal_scalable_ADM_df,
                                                          Tonal_scalable_ADM_means_df, PCA_df, PCA_means_df)

            cg_value = evaluation_cg(cloud_artist, radio, related_artists_df)

            if (alg, T) not in Radio_results.keys():
                Radio_results[(alg, T)] = cg_value
            else:
                Radio_results[(alg, T)] += cg_value

            gen_val_count = radio['genre1'].value_counts()
            for genre in gen_val_count.index:
                if alg not in Genres_dist.keys():
                    Genres_dist[alg] = {}

                if genre not in Genres_dist[alg].keys():
                    Genres_dist[alg][genre] = gen_val_count[genre]
                else:
                    Genres_dist[alg][genre] += gen_val_count[genre]

            artist_val_count = radio['recording_artist'].value_counts()
            for artist in artist_val_count.index:
                if alg not in Artists_dist.keys():
                    Artists_dist[alg] = {}

                if artist not in Artists_dist[alg].keys():
                    Artists_dist[alg][artist] = artist_val_count[artist]
                else:
                    Artists_dist[alg][artist] += artist_val_count[artist]

            if radio_flag:
                radio_flag = False
                # Options -  'PCA','DM','TSNE','SCL-ADM'
                if alg == 'SCL-ADM':
                    extended_scalable_ADM_df['radio_songs'] = 'not included'
                    extended_scalable_ADM_df['radio_songs'].iloc[radio_index] = 'in radio'
                    extended_scalable_ADM_df['radio_songs'].iloc[seed_index] = 'seed song'
                    full_tonal_data = extended_scalable_ADM_df
                    radio_tonal_data = extended_scalable_ADM_df[extended_scalable_ADM_df['radio_songs'] != 'not included']
                    out_of_radio_tonal_data = extended_scalable_ADM_df[extended_scalable_ADM_df['radio_songs'] == 'not included']
                elif alg == 'DM':
                    extended_DM_df['radio_songs'] = 'not included'
                    extended_DM_df['radio_songs'].iloc[radio_index] = 'in radio'
                    extended_DM_df['radio_songs'].iloc[seed_index] = 'seed song'
                    full_tonal_data = extended_DM_df
                elif alg == 'TSNE':
                    extended_TSNE_df['radio_songs'] = 'not included'
                    extended_TSNE_df['radio_songs'].iloc[radio_index] = 'in radio'
                    extended_TSNE_df['radio_songs'].iloc[seed_index] = 'seed song'
                    full_tonal_data = extended_TSNE_df
                else:
                    extended_PCA_df['radio_songs'] = 'not included'
                    extended_PCA_df['radio_songs'].iloc[radio_index] = 'in radio'
                    extended_PCA_df['radio_songs'].iloc[seed_index] = 'seed song'
                    full_tonal_data = extended_PCA_df

                # #Show all the songs which not included in the radio - low opacity
                # fig = px.scatter(out_of_radio_tonal_data, x='0', y='1', color='radio_songs', opacity = 0.5,
                #                  hover_data=['recording_artist', 'recording_name', 'genre1'])
                #
                # #Show only radio songs - bold
                # fig = px.scatter(radio_tonal_data, x='0', y='1', color='radio_songs',
                #                  color_discrete_sequence=px.colors.qualitative. Antique,
                #                  hover_data=['recording_artist', 'recording_name', 'genre1'])
                #
                # fig.update_layout(
                #     title='Radio songs - {} - based on {}, Tempo state - {} '.format(alg, cloud_artist, T))
                #
                # fig.show()


                # fig = px.scatter(full_tonal_data, x='0', y='1', color='radio_songs',
                #                      color_discrete_sequence=px.colors.qualitative. D3,
                #                  hover_data=['recording_artist', 'recording_name', 'genre1'])
                # fig.update_layout(title='Radio songs - {} - based on {}, Tempo state - {} '.format(alg, cloud_artist,T))
                # fig.show()


                ####  3D Visualization ####
                # fig = px.scatter_3d(full_tonal_data, x='0', y='1', z='2', color='radio_songs',
                #                     hover_data=['recording_artist', 'recording_name', 'genre1'])
                # fig.update_layout(title='Radio songs - {} - based on {}, Tempo state - {} '.format(alg, cloud_artist,T))
                # fig.show()

            print('Radio ' + str(radios_count) + ' complete')
            radios_count += 1

logging.debug('Finish creating radios')

logging.debug('Start Convert sum cg to avg cg')

# Convert sum value of cg to avg
for key in Radio_results.keys():
    Radio_results[key] = Radio_results[key] / num_of_iterations_for_each_setting
logging.debug('Finish Convert sum cg to avg cg')

# logging.debug('Start export radios results')
# a_file = open(
#     r"C:\Users\raphs\Documents\University\msc\thesis\data\Radios_results\{}_radios_results.pkl".format(cloud_artist),
#     "wb")
# pickle.dump(Radio_results, a_file)
# a_file.close()
# logging.debug('Finish export radios results')
#
# logging.debug('Start export Genres distribution')
# a_file = open(
#     r"C:\Users\raphs\Documents\University\msc\thesis\data\Radios_results\{}_genres dist.pkl".format(cloud_artist), "wb")
# pickle.dump(Genres_dist, a_file)
# a_file.close()
# logging.debug('Finish export Genres distribution')
#
# logging.debug('Start export Artists distribution')
# a_file = open(
#     r"C:\Users\raphs\Documents\University\msc\thesis\data\Radios_results\{}_artists dist.pkl".format(cloud_artist),
#     "wb")
# pickle.dump(Artists_dist, a_file)
# a_file.close()
# logging.debug('Finish export Artists distribution')

print(Radio_results)
