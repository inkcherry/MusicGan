global_config={
#sys config
'traindata_filename': "/home/liumingzhi/projectfloder/traindata/traindata.npz",
'data_shape':[4,48,84,5],    #[n_bars,n_timesteps_inbar,n_pitches,n_tracks]    Automate it in the future


#train config
'shuffle_size':1000,
'prefetch_size':1,
'batch_size':64,

#sampling
'sample_grid': [8,8],
'latent_dim':128

}

