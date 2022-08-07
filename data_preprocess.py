import pandas as pd
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

def preprocess(tonal_data,rhythm_data):
    logging.debug('Start data preprocess')
    ########### Pre-process ###########
    mfcc = tonal_data.iloc[:,79:92]

    # Option 2 - for 100,000 data - runtime check
    # mfcc = tonal_data.iloc[:,4:17]

    #Remove first column
    mfcc = mfcc.iloc[:,1:]

    #Normalize the data
    mfcc['sum']  = abs(mfcc.sum(axis=1))
    mfcc = mfcc.div(mfcc["sum"], axis=0)
    mfcc = mfcc.iloc[:,:-1] # Remove sum colunmn
    #### Weighting the columns of MFCC
    for i in range(mfcc.shape[1]):
        mfcc['lowlevel.mfcc.mean.[{}]$float'.format(str(i+1))] = (0.95**i) * mfcc['lowlevel.mfcc.mean.[{}]$float'.format(str(i+1))]


    # Quantile outlier removal by artist - big data

    artists = tonal_data['recording_artist'].unique()
    clean_mfccs = []
    clean_samples_index = []

    #Remove outliers by artist - remove 0.8 quantiles (20 upper percentage)
    for artist in artists:
        current_artist_mfcc = mfcc.loc[tonal_data['recording_artist'].str.contains(artist)].copy()
        current_clean_mfcc = current_artist_mfcc[current_artist_mfcc['lowlevel.mfcc.mean.[1]$float']<current_artist_mfcc['lowlevel.mfcc.mean.[1]$float'].quantile(0.8)].copy()
        clean_samples_index.append(current_clean_mfcc.index.values.tolist())
        clean_mfccs.append(current_clean_mfcc)
    mfcc_clean = pd.concat(clean_mfccs)

    flat_index = [item for sublist in clean_samples_index for item in sublist]
    tonal_data = tonal_data.iloc[flat_index].copy()

    #Add 'sample_location' column - using it when getting knn while creating playlist
    tonal_data['sample_location'] = 0

    for i in range(tonal_data.shape[0]):
        tonal_data['sample_location'].iloc[i] = i

    #Remove naan columns in rhythm data
    filter_int_columns_except_beats_count = [col for col in rhythm_data if col.endswith('int') if not col.startswith('rhythm.beats_count')]
    rhythm_data = rhythm_data.drop(filter_int_columns_except_beats_count,axis = 1)
    rhythm_data = rhythm_data.drop('rhythm.beats_loudness.min$float',axis = 1)
    rhythm_data = rhythm_data.loc[tonal_data.index,:]

    #Extract only numerical rhythm columns
    numerical_rhythm_data = rhythm_data.iloc[:,6:-3]

    #Drop naan from rhythm
    numerical_rhythm_data = numerical_rhythm_data.dropna()
    rhythm_data = rhythm_data.loc[numerical_rhythm_data.index,:]

    #Get data for TEMPO measure
    bpm_onset = rhythm_data.iloc[:,6:8]

    #clean the same samples from tonal data
    tonal_data = tonal_data.loc[numerical_rhythm_data.index,:]
    mfcc_clean = mfcc_clean.loc[numerical_rhythm_data.index,:]

    logging.debug('Finish data preprocess')
    return(mfcc_clean, tonal_data, rhythm_data, bpm_onset)

