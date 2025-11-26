To estimate bending persistence length based on the segment from base-pair 1 to 10:

import elasticity as elast

elast.bending_persistence(1,10, trajectory_name, topology_name)


#note that this calls function from bubble_parameter, and writes out files bending_angle_xx.dat


#note that this script requires the pytraj module to read in trajectory and topology files
