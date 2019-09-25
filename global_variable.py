global_config={
#sys config
'traindata_filename': "/home/liumingzhi/projectfloder/traindata/traindata.npz",
'data_shape':[4,48,84,5],    #[n_bars,n_timesteps_inbar,n_pitches,n_tracks]    Automate it in the future

'beat_resolution': 12,
#train config
'shuffle_size':1000,
'prefetch_size':1,
'batch_size':64,

'initial_learning_rate': 0.001,


'learning_rate_schedule':{
  'start': 45000,
  'end': 50000,
  'end_value': 0.0},

'adam':{
  'beta1': 0.5,
  'beta2': 0.9,
},
'steps': 50000,

'slope_schedule':{
  'end_value': 5.0,
  'start': 10000,
  'end': None
},

#sampling
'sample_grid': [8,8],
'latent_dim':128,


#save
'result_dir':"/home/liumingzhi/projectfloder/result"
}

