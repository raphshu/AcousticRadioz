import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_radio(extended_Tonal_ADM_df, final_radio, seed_songs_index, Algorithm, cloud_artist, number_of_tonal_clusters):
    #VISUALIZE
    extended_Tonal_ADM_df['radio_songs'] = 'not included'
    extended_Tonal_ADM_df['radio_songs'].loc[final_radio.index] = 'in radio'
    extended_Tonal_ADM_df['radio_songs'].loc[seed_songs_index] = 'seed song'

    # Use instead of tonal_data to color only the samples the clouds based on -> cloud_clusters = tonal_data[tonal_data['kmeans_result']<9].copy()
    # Options - ADM, DM, TSNE
    if Algorithm == 'ADM':
        full_tonal_data = extended_Tonal_ADM_df
    # elif Algorithm == 'DM':
    #     full_tonal_data = extended_DM_df
    # else:
    #     full_tonal_data = extended_TSNE

    fig = px.scatter(full_tonal_data, x=0, y=1, color = 'radio_songs',hover_data = ['recording_artist','recording_name','genre1'])
    fig.update_layout(title='MFCC {} - based on {} - {} groups'.format(Algorithm,cloud_artist,number_of_tonal_clusters))
    fig.show()

    fig = px.scatter_3d(full_tonal_data, x=0, y=1,z=2, color = 'radio_songs',hover_data = ['recording_artist','recording_name','genre1'])
    fig.update_layout(title='MFCC {} - based on {} - {} groups'.format(Algorithm,cloud_artist,number_of_tonal_clusters))
    fig.show()