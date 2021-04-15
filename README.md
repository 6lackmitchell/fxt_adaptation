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
This repository uses ROS-Noetic with Anaconda (Python 3) on Ubuntu 20.04.2 LTS. For ROS installation instructions, go to http://wiki.ros.org/noetic/Installation/Ubuntu. This repository also depends on the following:</br>

https://github.com/ethz-asl/asctec_mav_framework (which contains its own dependencies -- follow instructions there)
https://github.com/ethz-asl/mav_comm </br>
https://github.com/ethz-asl/vicon_bridge (at time of writing the required repository was https://github.com/lukasfro/vicon_bridge, which had fixed a bug related to the Boost version in Ubuntu 20.04, but it is possible that the ETH team will have merged this fix by time of reading)</br>

We recommend creating a separate ~/git repository in which to install these, and then symbolically linking the git repos to the catkin workspace (i.e. ln -s /home/user/git/repo /home/user/catkin_ws/src). Prior to building, there are several bugs which need to be addressed. First, edit the ~/git/asctec_mav_framework/asctec_hl_interface/CMakeList.txt file such that the find_package function includes "mav_msgs" as an argument (we placed it directly below "geometry_msgs"). Second, open ~/git/asctec_mav_framework/asctec_hl_interface/src/comm.cpp and change the uint32_t designation before baudrate in lines 51, 72, and 118 to int32_t. Make the corresponding changes to the function declarations inside the ~/git/asctec_mav_framework/asctec_hl_interface/src/comm.h file as well. Then, in that same file, change the "static const" type identifier in line 99 to "static constexpr". This should allow you to successfully run "catkin build" when the time comes.

1. Follow [ros/catkin instructions](http://wiki.ros.org/catkin/Tutorials/create_a_workspace) to build the workspace (~/catkin_ws).
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

