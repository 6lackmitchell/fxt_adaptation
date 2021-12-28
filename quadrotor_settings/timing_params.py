# Timing variables
ts            = 0.0  # Start time (sec)
tf            = 15.0 # End time (sec)
#tf            = 2.50 # End time (sec)
dt            = 1e-4 # timestep (sec)
#dt            = 1e-6 # timestep (sec)
#dt            = 1e-3 # timestep (sec)
nTimesteps    = int((tf - ts) / dt) + 1