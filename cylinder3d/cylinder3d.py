import lettuce as lt
import torch
import numpy as np
import sys

class NaNReporter:
    """Reports any NaN and aborts the simulation"""
    def __call__(self,i,t,f):
        if torch.isnan(f).any()==True:
            print ("NaN detected in time step ", i)
            print ("Abort")
            sys.exit()

class AveragedVelocityReporter:
    """Reports the streamwise velocity averaged in span direction (z) at x=x_row"""
    def __init__(self, lattice, flow, x_row, interval=1, starting_step=1, out=None):
        self.lattice = lattice
        self.flow = flow
        self.x_row = x_row
        self.interval = interval
        self.starting_step = starting_step
        self.out = [] if out is None else out

    def __call__(self, i, t, f):
        u = self.lattice.u(f)[:,self.x_row,:,:]
        u = self.flow.units.convert_velocity_to_pu(u).cpu().numpy()
        entry = np.mean(u,axis=2)
        if isinstance(self.out, list):
            self.out.append(entry)
        else:
            print(*entry, file=self.out)


# "cuda:0" for GPU
device=torch.device("cuda:0")
stencil = lt.D3Q27
lattice = lt.Lattice(stencil,device=device,dtype=torch.float32)

#Time period of the simulation
max_t=70
#Start recording the averaged velocity value at time
time_start_recording = 35


#Diameter of the cylinder
D = 36
grid_points_per_D_in_X = 14
grid_points_per_D_in_Y = 10
grid_points_per_D_in_Z = 3
flow=lt.Obstacle3D(grid_points_per_D_in_X*(D),grid_points_per_D_in_Y*(D),grid_points_per_D_in_Z*(D),reynolds_number=3900,mach_number=0.075,lattice=lattice,char_length_lu=(D))

#Create a mask to determine the bounce back boundary of the cylinder
x = flow.grid
mask_np = np.zeros([flow.resolution_x,flow.resolution_y,flow.resolution_z],dtype=bool)

#The center of the cylinder is located at the following coordinates
center_x = 5*D
center_y = int(grid_points_per_D_in_Y/2*D)

# Set the mask for the cylinder (Note that bounce back boundaries set the wall between the fluid and the boundary node)
for X in np.arange(flow.resolution_x):
    for Y in np.arange(flow.resolution_y):
        if ((X-center_x)**2 + (Y-center_y)**2) <= (((D-1)/2)**2):
            mask_np[X,Y,:] = True

boundary_points_mask = np.zeros_like(mask_np)
boundary_angle_info = np.ones_like(mask_np,dtype=float)*1000

flow.mask=mask_np
collision=lt.KBCCollision3D(lattice, tau=flow.units.relaxation_parameter_lu)
streaming=lt.StandardStreaming(lattice)
lattice.equilibrium = lt.QuadraticEquilibrium_LessMemory(lattice)

simulation=lt.Simulation(flow,lattice,collision,streaming)

#Create and append the NaN reporter to detect instabilities
NaN=NaNReporter()
simulation.reporters.append(NaN)

#Create the averaged velocity reporter at three different locations and append to reporters (start at t=35)
Velocity0= AveragedVelocityReporter(lattice, flow, int(round(center_x + D * 1.06, 0)), 1, int(flow.units.convert_time_to_lu(time_start_recording)))
Velocity1= AveragedVelocityReporter(lattice, flow, int(round(center_x + D * 1.54, 0)), 1, int(flow.units.convert_time_to_lu(time_start_recording)))
Velocity2= AveragedVelocityReporter(lattice, flow, int(round(center_x + D * 2.02, 0)), 1, int(flow.units.convert_time_to_lu(time_start_recording)))

simulation.reporters.append(Velocity0)
simulation.reporters.append(Velocity1)
simulation.reporters.append(Velocity2)

#If desired append a VTK reporter, reporting every 7500 simulation steps
vtk_rep = lt.VTKReporter(lattice,flow,7500,'./data/cylinder')
simulation.reporters.append((vtk_rep))


print ("Simulating steps:", int(flow.units.convert_time_to_lu(max_t)))
simulation.step(int(flow.units.convert_time_to_lu(max_t)))

# Set a checkpoint for further investigations
simulation.save_checkpoint('checkpoint')
#simulation.load_checkpoint('checkpoint')

#Postprocessing

# Save the velocities as numpy arrays for further post processing
np.save("Vel0.npy",np.array(Velocity0.out))
np.save("Vel1.npy",np.array(Velocity1.out))
np.save("Vel2.npy",np.array(Velocity2.out))
