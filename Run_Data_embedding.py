import pandas as pd
import logging
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import datetime
import json
import pickle

from data_preprocess import preprocess
from ADM_cloud_creation import create_clouds
from ADM import adm
from ADM_results_post_process import ADM_post_process
from SCL_ADM import scalable_adm
from SCL_ADM_results_post_process import  SCL_ADM_post_process
from DM import diffusionMapping
from PCA_results_post_process import PCA_post_process
from DM_results_post_process import DM_post_process
from TSNE_results_post_process import TSNE_post_process
from Visualize_the_radio import plot_radio
pd.set_option("max_rows", None)
pd.set_option('display.max_columns', None)


########### Import the data ###########

#### Data for runtime checks - option 1 ####
# tonal_data = pd.read_csv(r'C:\Users\raphs\Documents\University\msc\thesis\data\acoustic_brainz_genre_set\All_features\genreset_100000_tonal.csv')
# rhythm_data = pd.read_csv(r'C:\Users\raphs\Documents\University\msc\thesis\data\acoustic_brainz_genre_set\All_features\genreset_100000_rhythm.csv')
# tonal_data = tonal_data.iloc[:70000,:]
# rhythm_data = rhythm_data.iloc[:70000,:]


#Data for runtime checks - option 2 - duplicate the main study data
# tonal_data = pd.concat([tonal_data,tonal_data,tonal_data])
# rhythm_data = pd.concat([rhythm_data,rhythm_data,rhythm_data])
# tonal_data = tonal_data.reset_index(drop = True)
# rhythm_data = rhythm_data.reset_index(drop = True)


#Main study data
tonal_data = pd.read_csv(r'C:\Users\raphs\Documents\University\msc\thesis\data\acoustic_brainz_genre_set\All_features\genreset_extended_songs_and_features_tonal.csv')
rhythm_data = pd.read_csv(r'C:\Users\raphs\Documents\University\msc\thesis\data\acoustic_brainz_genre_set\All_features\genreset_extended_songs_and_features_rhythm.csv')

###########  Pre-procss ###########
mfcc_clean, tonal_data, rhythm_data, bpm_onset = preprocess(tonal_data, rhythm_data)

## ADM ##

########### Set and mark on data - cloud artist ###########
str_time = datetime.datetime.now()

cloud_artist = 'the beatles'
tonal_data.loc[tonal_data['recording_artist'].str.contains(cloud_artist), 'is_cloud'] = 1
tonal_data.loc[tonal_data['recording_artist'].str.contains(cloud_artist) == False, 'is_cloud'] = 0


########## ADM(scl) ###########
#Slice cloud artist data
cloud_mfcc = mfcc_clean.loc[tonal_data['recording_artist'].str.contains(cloud_artist)].copy()
number_of_tonal_clusters = min(15,cloud_mfcc.shape[0])

########### Set epsilon and Number of dimensions in result ###########
eps = 0.05 / (450/(cloud_mfcc.shape[0]))
print('epsilon factor: '+str(eps))

embedding_dim = 4

#time stamp
str_time = datetime.datetime.now()

#Create the clouds
tonal_data, cloud_mfcc, central_points, clusters_cov_inverse = create_clouds(tonal_data, cloud_mfcc,
                                                                             number_of_tonal_clusters)

#Run- REF-DM
tonal_scalable_adm_mfcc = scalable_adm(mfcc_clean,central_points,clusters_cov_inverse,
                                       mfcc_clean.shape[0],number_of_tonal_clusters,eps,embedding_dim)
#time stamp
fnsh_time = datetime.datetime.now()

#Save scl_adm runtime
scl_adm_time = fnsh_time - str_time
scl_adm_run_time = {'samples': mfcc_clean.shape[0],'method': 'SCL_ADM', 'time':scl_adm_time}

#Set SCL_ADM-results for playlist and visualize
tonal_data, rhythm_data, Tonal_scalable_ADM_df, Tonal_scalable_ADM_means_df, extended_scalable_ADM_df,\
artist_genre_df= SCL_ADM_post_process(tonal_scalable_adm_mfcc, tonal_data, rhythm_data)

########### DM ###########
#time stamp
str_time = datetime.datetime.now()

#Run
tonal_dm_mfcc = diffusionMapping(mfcc_clean,1,dim=embedding_dim)

#time stamp
fnsh_time = datetime.datetime.now()

#Save dm runtime
dm_time = fnsh_time - str_time
dm_run_time = {'samples': mfcc_clean.shape[0],'method': 'DM', 'time':dm_time}

#Post process
DM_df, extended_DM_df, DM_means_df, rhythm_info = DM_post_process(tonal_dm_mfcc, tonal_data, rhythm_data)

########## PCA ###########

#time stamp
str_time = datetime.datetime.now()

#Run
logging.debug('Start PCA')
tonal_pca_mfcc = PCA(n_components = (embedding_dim -1)).fit_transform(mfcc_clean)
logging.debug('Finish PCA')

#time stamp
fnsh_time = datetime.datetime.now()

#Save pca runtime
pca_time = fnsh_time - str_time
pca_run_time = {'samples': mfcc_clean.shape[0],'method': 'PCA', 'time':pca_time}

#Post process
PCA_df, extended_PCA_df, PCA_means_df, rhythm_info = PCA_post_process(tonal_pca_mfcc, tonal_data, rhythm_data)

########### TSNE ###########

#time stamp
str_time = datetime.datetime.now()

#Run
logging.debug('Start TSNE')
Tsne_result = TSNE(n_components = (embedding_dim -1)).fit_transform(mfcc_clean)
logging.debug('Finish TSNE')

#time stamp
fnsh_time = datetime.datetime.now()

#Save TSNE runtime
tsne_time = fnsh_time - str_time
TSNE_run_time = {'samples': mfcc_clean.shape[0],'method': 'TSNE', 'time':tsne_time}

#Post process
extended_TSNE_df, TSNE_df, TSNE_means_df = TSNE_post_process(Tsne_result, tonal_data, rhythm_data)

########## Export run time ###########

logging.debug('Start export run time json')

try:
    a = pickle.load(open("run_times.pickle","rb"))
except:
    a = []

a.append(scl_adm_run_time)
a.append(dm_run_time)
a.append(pca_run_time)
a.append(TSNE_run_time)
#
with open('run_times.pickle','wb') as fp:
    pickle.dump(a, fp, protocol=pickle.HIGHEST_PROTOCOL)

logging.debug('Finish export run time json')

Export data for Radios creation and comparison
tonal_data.to_csv(r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_tonal_data.csv".format(cloud_artist))
bpm_onset.to_csv(r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_bpm_onset.csv".format(cloud_artist))

Tonal_ADM_df.to_csv(r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_Tonal_ADM_df.csv".format(cloud_artist))
Tonal_ADM_means_df.to_csv(r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_Tonal_ADM_means_df.csv".format(cloud_artist))
Tonal_scalable_ADM_df.to_csv(r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_Tonal_scalable_ADM_df.csv".format(cloud_artist))
Tonal_scalable_ADM_means_df.to_csv(r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_Tonal_scalable_ADM_means_df.csv".format(cloud_artist))
DM_df.to_csv(r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_DM_df.csv".format(cloud_artist))
DM_means_df.to_csv(r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_DM_means_df.csv".format(cloud_artist))
PCA_df.to_csv(r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_PCA_df.csv".format(cloud_artist))
PCA_means_df.to_csv(r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_PCA_means_df.csv".format(cloud_artist))
TSNE_df.to_csv(r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_TSNE_df.csv".format(cloud_artist))
TSNE_means_df.to_csv(r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_TSNE_means_df.csv".format(cloud_artist))

adm_features = [number_of_tonal_clusters, cloud_artist]

#Export data for Radio visualization
extended_scalable_ADM_df.to_csv(r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_extended_scalable_ADM_df.csv".format(cloud_artist))
extended_DM_df.to_csv(r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_extended_DM_df.csv".format(cloud_artist))
extended_PCA_df.to_csv(r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_extended_PCA_df.csv".format(cloud_artist))
extended_TSNE_df.to_csv(r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_extended_TSNE_df.csv".format(cloud_artist))

with open(r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\adm_features.csv", "w") as output:
    output.write(str(adm_features))