import pandas as pd
import plotly.express as px
import pickle
cloud_artist = 'ludwig van beethoven'
infile = open(r"C:\Users\raphs\Documents\University\msc\thesis\data\Radios_results\{}_genres dist.pkl".format(cloud_artist),'rb')
genres_dist = pickle.load(infile)
infile.close()

infile = open(r"C:\Users\raphs\Documents\University\msc\thesis\data\Radios_results\{}_artists dist.pkl".format(cloud_artist),'rb')
artists_dist = pickle.load(infile)
infile.close()

infile = open(r"C:\Users\raphs\Documents\University\msc\thesis\data\Radios_results\{}_radios_results.pkl".format(cloud_artist),'rb')
Radios_result = pickle.load(infile)
infile.close()

a = {}

for i in Radios_result.keys():
    alg = i[0]
    tempo = i[1]
    if alg not in a.keys():
        a[alg] = {}

    a[alg][tempo] = Radios_result[i]
    results_df = pd.DataFrame(a).T
    results_df = results_df.sort_values(by = True, ascending= False)

print("Average CG score of 10 iterations for each setting - based on {}".format(cloud_artist))
print(results_df)
print('-----------------------')


#Show genre distribution per algorithm
for alg in genres_dist.keys():
    genres_dist_df = pd.DataFrame.from_dict(genres_dist[alg], orient='index')
    genres_dist_df = genres_dist_df.reset_index()
    genres_dist_df = genres_dist_df.rename(columns={"index": "Genre", 0: "number of songs"})
    genres_dist_df = genres_dist_df.sort_values(by = 'number of songs', ascending= False)
    fig = px.bar(genres_dist_df, x='Genre', y='number of songs', hover_data = ['number of songs'])
    fig.update_layout(title='Genres distribution -{} Algorithm based on {}'.format(alg,cloud_artist))
    fig.show()

# #Show artists distribution
#
# for alg in artists_dist.keys():
#     artists_dist_df = pd.DataFrame.from_dict(artists_dist[alg], orient='index')
#     artists_dist_df = artists_dist_df.reset_index()
#     artists_dist_df = artists_dist_df.rename(columns={"index": "Artist", 0: "number of songs"})
#     artists_dist_df = artists_dist_df.sort_values(by = 'number of songs', ascending= False)
#     fig = px.bar(artists_dist_df, x='Artist', y='number of songs', hover_data = ['number of songs'])
#     fig.update_layout(title='Artists distribution -{} Algorithm based on {}'.format(alg,cloud_artist))
#     fig.show()

