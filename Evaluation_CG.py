import pandas as pd
pd.set_option('display.max_colwidth', None)


def evaluation_cg(artist, final_radio, related_artists_df):

    #Remove artists with less than 4 common songs
    related_artists_df = related_artists_df[related_artists_df['Joint appearance']>3]

    related_artists_df['Relevance'] = 0

    related_artists_df['Relevance'][related_artists_df['Joint appearance'] > related_artists_df['Joint appearance'].quantile(0.99)] = 5
    related_artists_df['Relevance'][(related_artists_df['Joint appearance'] > related_artists_df['Joint appearance'].quantile(0.75)) &
                                    (related_artists_df['Joint appearance'] <= related_artists_df['Joint appearance'].quantile(0.99))] = 4
    related_artists_df['Relevance'][(related_artists_df['Joint appearance'] > related_artists_df['Joint appearance'].quantile(0.55)) &
                                    (related_artists_df['Joint appearance'] <= related_artists_df['Joint appearance'].quantile(0.75))] = 3
    related_artists_df['Relevance'][(related_artists_df['Joint appearance'] > related_artists_df['Joint appearance'].quantile(0.35)) &
                                    (related_artists_df['Joint appearance'] <= related_artists_df['Joint appearance'].quantile(0.55))] = 2
    related_artists_df['Relevance'][related_artists_df['Joint appearance']<=related_artists_df['Joint appearance'].quantile(0.35)] = 1

    final_radio = final_radio.rename(columns={"recording_artist": "Artist"})
    full_eval = pd.merge(final_radio, related_artists_df, how="left", on='Artist')

    #Finll Naan - Set artists that doesn't have any common songs with beatles as 0
    is_NaN = full_eval.isnull()
    row_has_NaN = is_NaN.any(axis=1)
    rows_with_NaN = full_eval[row_has_NaN]
    full_eval['Relevance'].loc[rows_with_NaN.index] = 0
    cg_value = full_eval['Relevance'].sum()
    return(cg_value)