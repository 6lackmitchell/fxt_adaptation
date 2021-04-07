# Determine which problem is to be simulated
import sys
args = sys.argv
if len(args) > 1:
    config = str(args[1])
else:
    config = 'simple'

if config == 'simple':
    import viz.simple_vis
if config == 'simple2ndorder':
    import viz.simple2ndorder_vis
elif config == 'overtake':
    import viz.overtake_vis
elif config == 'quadrotor':
    import viz.quadrotor_vis
