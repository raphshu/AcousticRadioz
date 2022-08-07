# #Run-ADM
# tonal_adm_mfcc = adm(mfcc_clean,central_points,clusters_cov_inverse,mfcc_clean.shape[0],
#                      number_of_tonal_clusters,eps,embedding_dim)
# #time stamp
# fnsh_time = datetime.datetime.now()
#
# #Save adm runtime
# adm_time = fnsh_time - str_time
# adm_run_time = {'samples': mfcc_clean.shape[0],'method': 'ADM', 'time':adm_time}
#
# #Set ADM-results for playlist and visualize
# tonal_data, rhythm_data, Tonal_ADM_df, Tonal_ADM_means_df, extended_Tonal_ADM_df,\
# artist_genre_df= ADM_post_process(tonal_adm_mfcc, tonal_data, rhythm_data)