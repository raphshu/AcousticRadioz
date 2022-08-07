# AcousticRadioz
AcousticRadioz source code - implementations of embedding methods and automatic playlist generation based on AcousticBrainz data

Work flow to create a radio - 

1.	Run ‘Run_Data_Embedding.py’ to create the embedded data by all the four methods, while paying attention to the Artist you want to base the RED-DM on. 
•	Make sure you have Related_artists_df from MELON file of the designated cloud artist for the evaluation process.

•	If not – create it with the Jupyter notebook – ‘Melon playlists.ipynb’ based in ‘Import data to csv’ folder. 
2.	Run ‘Run_and_compare_radios.py’ while paying attention to the cloud artist you choose. 
•	The code will take the right embedded data based on the cloud artist you choose. 
•	Pay attention to the name of exported code – No divide/Regular. No divide means that part of the seed artist songs were included in the base clouds and others didn’t. 
3.	Run ‘Run_visualize_radio_results.py’ and pay attention to cloud_artist in the beginning. 

