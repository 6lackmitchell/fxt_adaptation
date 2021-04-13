# fxt_adaptation
> This project seeks to reduce conservatism in safety-critical controllers by learning the unknown parameters in a parameter-affine controlled dynamical system within a fixed-time (FxT) independent of the initial parameter estimates. The theoretical results of this research were developed as a collaborative effort between myself, my advisor Dimitra Panagou, and Dr. Ehsan Arabi of Ford Motor Company. The provided code demonstrates the viability of our theoretical contributions both in simulation and via hardware experimentation with case studies on a quadrotor in a wind field.
<hr>

# Table of Contents
* [Team Members](#team-members)</br>
* [Simulation](#simulation)</br>
* [Experimentation](#experimentation)

# <a name="team-members"></a>Team Members
* "Mitchell Black" <mblackjr@umich.edu>
* "Ehsan Arabi" <earabi@umich.edu>
* "Dimitra Panagou" <dpanagou@umich.edu>

# <a name="simulation"></a>Simulation
Something here about the quadrotor simulation in Python

# <a name="experimentation"></a>Experimentation
### General Notes
* All python executable files must begin with the "shebang", i.e. "#!/usr/bin/env python" or specifically the location of your desired python executable
### Setup
1. Follow [ros/catkin instructions](http://wiki.ros.org/catkin/Tutorials/create_a_workspace) to build the workspace.
2. Start the ROS Core service. Navigate the your workspace, source the appropriate setup.bash file (e.g. source devel/setup.bash), and run
> roscore
3. Start the Vicon Bridge service. Open a new terminal, source the setup.bash file again, and run
> roslaunch vicon_bridge vicon.launch
4. Start the XBee communication. Open a new terminal, source the setup file again, and run
> roslaunch asctec_hl_interface quadx.launch
5. It is at this point where William starts MATLAB. Here, maybe I will need to start Python/ROS wrapper.
6. Start the main program. Open a new terminal, source the setup file again, and run
> rosrun <pkg_name> main
William's main program is written in C++ and it commands the quadrotor to take off and hover at a given height, and then it essentially switches over to a MATLAB program to execute the control algorithm.

