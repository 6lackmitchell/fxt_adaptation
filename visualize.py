# Determine which problem is to be simulated
import sys
args = sys.argv
file_provided = False
if len(args) > 1:
    config = str(args[1])
    if len(args) > 2:
        import builtins
        builtins.VIS_FILE = str(args[2])
        file_provided = True
else:
    config = 'simple'

if config == 'simple':
    import viz.simple_vis
if config == 'simple2ndorder':
    import viz.simple2ndorder_vis
elif config == 'overtake':
    import viz.overtake_vis
elif config == 'quadrotor':
    if file_provided:
        import viz.quadrotor_vis
    else:
        import viz.quadrotor_vis_casestudy2

