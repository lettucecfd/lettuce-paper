import torch
import numpy as np
from lettuce import (
    StandardStreaming, RegularizedCollision, BGKCollision, KBCCollision2D,
    DoublyPeriodicShear2D, Lattice, DecayingTurbulence, TaylorGreenVortex2D,
    D2Q9, Simulation, D2Q9Dellar, grid_fine_to_coarse, MRTCollision, EnergySpectrum
)

from lettuce import Enstrophy, IncompressibleKineticEnergy, ObservableReporter
import matplotlib.pyplot as plt
import sys
from lettuce.util import torch_gradient
from torch.optim.lr_scheduler import StepLR

########################################################################################################################
##### Parameter
########################################################################################################################
# Set flag, when running on a cluster. Backend hat to be changed, otherwise visualisation problems occurs
cluster = True if len(sys.argv) > 1 else False
if cluster:
    import matplotlib
    matplotlib.use('Agg')

### Hardware
device = "cuda"
train = False
filebase = "./net_preset.pth"


########################################################################################################################
##### Parameter
########################################################################################################################
data_interval = 100  # number of simulation steps between start and finish of each sample
n_steps = 4000      # total number of simulation steps for generating "groud truth" data
reynolds = 5000
mach = 0.05
resolution = 64

nr_init = 0

### Training parameter
n_epochs = 50
print_interval = 25
learning_rate = 3e-4
schedule = True
schedule_step_size = 40
schedule_step_gamma = 0.1

flowclass = DoublyPeriodicShear2D

postplotinterval = 5

### Training parameter
SaveNet = True
SaveFig = True

### Additional variables for 'DecayingTurbulence'
k0 = 9 # Defines spectrum peak.
seed = 0 # Defines constant random Gaussian distribution
cmap = 'seismic'
ic_energy = 0.5

### Additional variables for script
PATH = "_"+str(sys.argv[1]) if len(sys.argv) > 1 else "_local" # Index on workstation: '_local_; on cluster: ' #jobnumber'
ratio = 0.8 # Splits available reference data in training and testing data
dataset_training = int(n_steps/data_interval*ratio)
dataset_testing = int(n_steps/data_interval*(1-ratio))


########################################################################################################################
##### Data reporter
########################################################################################################################
class FReporter:
    def __init__(self, interval):
        self.interval = interval
        self.fs = []

    def __call__(self, i, t, f):
        if i % self.interval == 0:
            self.fs.append(grid_fine_to_coarse(lattice, f.clone(), tau_fine, tau_coarse).cpu())

class FReporter_coarse:
    def __init__(self, interval):
        self.interval = interval
        self.fs = []
        self.ts = []

    def __call__(self, i, t, f):
        if i % self.interval == 0:
            self.fs.append(f)
            self.ts.append(i)

def vorticity(f):
    u = lattice.u(f)
    grad_u0 = torch_gradient(u[0], 1)
    grad_u1 = torch_gradient(u[1], 1)
    return (grad_u1[0] - grad_u0[1])
########################################################################################################################
##### Define loss function
########################################################################################################################
def loss_function(x,y):
    # return torch.sqrt((x-y)**2).sum()
    return ((x - y) ** 2).sum()

########################################################################################################################
##### Neuronal net
########################################################################################################################
class LearnedMRT(torch.nn.Module):
    def __init__(self, tau, moment_transform, activation=torch.nn.ReLU()):
        super().__init__()
        self.__name__ = "Learned collision"
        self.tau = tau
        self.trafo = moment_transform  # lt.DellarD2Q9
        # 1st net for higher order moment relaxation: Jx and Jy
        self.j_net = torch.nn.Sequential(
            torch.nn.Linear(9,24),
            activation,
            torch.nn.Linear(24,1)
        )
        # 2nd net for higher order moment relaxtion: N
        self.n_net = torch.nn.Sequential(
            torch.nn.Linear(9,24),
            activation,
            torch.nn.Linear(24,1)
        )
    def flip_xy(self, m):
        """flip x and y and moments"""
        assert self.trafo.__class__ == D2Q9Dellar  # other moment sets have different ordering of moments
        return m[:,:,[0,2,1,5,4,3,6,8,7]]
    @staticmethod
    def gt_half(a):
        """transform into a value > 0.5"""
        return 0.5 + torch.exp(a)
    def __call__(self, f):
        return self.forward(f)
    def forward(self, f):
        qdim, nxdim, nydim = f.shape
        # transform to moment space
        m = self.trafo.transform(f)
        mt = m.permute(1, 2, 0)  # grid dimensions are batch dims for the networks
        assert mt.shape == (nxdim, nydim, qdim)
        # determine higher-order moment relaxation parameters
        tau_jx = self.gt_half(self.j_net.forward(mt))
        tau_jy = self.gt_half(self.j_net.forward(self.flip_xy(mt)))
        tau_n = self.gt_half(self.n_net.forward(mt) + self.n_net.forward(self.flip_xy(mt)))
        # print(str(torch.min(tau_n))+"   "+str(torch.max(tau_n)))
        # by summing over xy-ordered and yx-ordered, we make tau_n rotation equivariant
        assert tau_jx.shape == (nxdim, nydim, 1)
        assert tau_jy.shape == (nxdim, nydim, 1)
        assert tau_n.shape == (nxdim, nydim, 1)
        # assign tau to moments
        taus = self.tau * torch.ones_like(m)
        taus[6] = tau_n[...,0]
        taus[7] = tau_jx[...,0]
        taus[8] = tau_jy[...,0]
        assert taus.shape == (qdim, nxdim, nydim)
        # relax
        meq = self.trafo.equilibrium(m)
        m_postcollision = m - 1./taus * (m - meq)
        return self.trafo.inverse_transform(m_postcollision)


########################################################################################################################
##### Setup
########################################################################################################################
class DecayingTurbulence_validation:

    def __init__(self, resolution, reynolds_number, mach_number, lattice, k0=20, ic_energy=0.5, case=None):
        from lettuce.unit import UnitConversion
        self.k0 = k0
        self.ic_energy = ic_energy
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=2 * np.pi,
            characteristic_velocity_pu=None
        )
        self.case = case
        self.dimensions = self.grid[0].shape

    def analytic_solution(self, x, t=0):
        return

    def _compute_initial_velocity(self):
        if self.case == 'coarse':
            u = np.load('InitialVelocity_decaying_coarse.npy')
        if self.case == 'fine':
            u = np.load('InitialVelocity_decaying_fine.npy')
        return u

    def _compute_initial_pressure(self):
        return np.zeros(self.dimensions)[None, ...]

    def initial_solution(self, x):
        """Return initial solution. Note: this function sets the characteristic velocity in phyiscal units."""
        u = self._compute_initial_velocity()
        p = self._compute_initial_pressure()
        self.units.characteristic_velocity_pu = np.linalg.norm(u, axis=0).max()
        return p, u

    @property
    def energy_spectrum(self):
        return self.spectrum, self.wavenumbers

    @property
    def grid(self):
        grid = [np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False) for _ in range(self.units.lattice.D)]
        return np.meshgrid(*grid)

    @property
    def boundaries(self):
        return []


print('Generates reference data on fine grid')
torch.cuda.empty_cache()

lattice = Lattice(D2Q9, device=device, dtype=torch.double)
flow_fine = flowclass(
    resolution=2*resolution,
    reynolds_number=reynolds,
    mach_number=mach,
    lattice=lattice,
)
tau_fine = flow_fine.units.relaxation_parameter_lu
flow_coarse = flowclass(
    resolution=resolution,
    reynolds_number=reynolds,
    mach_number=mach,
    lattice=lattice,
)


collision = BGKCollision(lattice, tau=tau_fine)
streaming = StandardStreaming(lattice)
simulation = Simulation(flow=flow_fine, lattice=lattice, collision=collision, streaming=streaming)

flow_coarse.units.characteristic_velocity_pu = flow_fine.units.characteristic_velocity_pu
tau_coarse = flow_coarse.units.relaxation_parameter_lu

### Reporter
spectrum = EnergySpectrum(lattice, flow_fine)
enstrophy_f = ObservableReporter(Enstrophy(lattice, flow_fine), interval=postplotinterval*2, out=None)
energy_f = ObservableReporter(IncompressibleKineticEnergy(lattice, flow_fine), interval=postplotinterval*2, out=None)
spectrum_f = ObservableReporter(EnergySpectrum(lattice, flow_fine), interval=postplotinterval*2, out=None)
f_reporter = FReporter(2*data_interval)
simulation.reporters.append(f_reporter)
simulation.reporters.append(enstrophy_f)
simulation.reporters.append(energy_f)
simulation.reporters.append(spectrum_f )

simulation.initialize_pressure()
simulation.initialize_f_neq()
_ = simulation.step(2*n_steps)


########################################################################################################################
##### Make Data Pairs
########################################################################################################################
print('Make Data Pairs and shuffle training data')
data = torch.stack(f_reporter.fs).cpu()
f_reporter.fs = []

inputs = data[0:-1]
outputs = data[1:]

f_init = inputs[nr_init].clone()
f_output = outputs[-1].clone()

shuffle = torch.randperm(inputs.shape[0])
inputs = inputs[shuffle]
outputs = outputs[shuffle]

########################################################################################################################
##### Setup Collision model based on neural network
########################################################################################################################
torch.autograd.set_detect_anomaly(True)
learned_mrt = LearnedMRT(tau=tau_coarse, moment_transform=D2Q9Dellar(lattice))
learned_mrt.to(device=lattice.device, dtype=lattice.dtype)

########################################################################################################################
##### Start training
########################################################################################################################
if train:
    def train(model, optimizer, inputs, outputs, simulation, energy=None, spectrum=None, device="cuda"):
        losses = []
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            optimizer.zero_grad()
            simulation.f = inp.clone().cuda()
            simulation.step(data_interval)
            loss = (loss_function(vorticity(simulation.f), vorticity(out.clone().cuda())) * 0.2
                    + loss_function(energy(simulation.f), energy(out.clone().cuda())) * 0.2
                    + loss_function(lattice.u(simulation.f), lattice.u(out.cuda())) * 0.6
                    )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return np.mean(losses)

    optimizer = torch.optim.Adam(learned_mrt.parameters(), lr=learning_rate)
    if schedule:
        scheduler = StepLR(optimizer, step_size=schedule_step_size, gamma=schedule_step_gamma)

    simulation_l = Simulation(flow=flow_coarse, lattice=lattice, collision=learned_mrt, streaming=streaming)
    energy = IncompressibleKineticEnergy(lattice, flow_coarse)
    enstrophy = Enstrophy(lattice, flow_coarse)
    spectrum_c = EnergySpectrum(lattice, flow_coarse)

    losses_training = []
    losses_testing = []
    print(f"Start training through {n_epochs} epochs")
    for epoch in range(n_epochs):
        learned_mrt.train(True)
        loss = train(model=learned_mrt, optimizer=optimizer, inputs=inputs[:dataset_training],
                           outputs=outputs[:dataset_training], simulation=simulation_l, energy=energy, spectrum=spectrum_c)
        losses_training.append(loss)
        learned_mrt.train(False)
        loss = train(model=learned_mrt, optimizer=optimizer, inputs=inputs[dataset_training:],
                           outputs=outputs[dataset_training:], simulation=simulation_l, energy=energy, spectrum=spectrum_c)
        losses_testing.append(loss)
        print(f"Epoch {epoch + 1}" + f" - Mean loss (training): {losses_training[-1]:2.6e} " + f"    Mean loss (testing): {losses_testing[-1]:2.6e} ")
        if schedule:
            scheduler.step()

    if SaveNet:
        torch.save(learned_mrt.state_dict(), "./net" + PATH + ".pth")

########################################################################################################################
##### Comparison with other collision models
########################################################################################################################
print('Comparison to other collision models')

if not train:
    learned_mrt.load_state_dict(torch.load(filebase))
    learned_mrt.eval()

energy_data = []
enstrophy_data = []
spectrum_data = []
f_image = []

energy = IncompressibleKineticEnergy(lattice, flow_coarse)
enstrophy = Enstrophy(lattice, flow_coarse)
spectrum_c = EnergySpectrum(lattice, flow_coarse)
enstrophy_r = ObservableReporter(Enstrophy(lattice, flow_coarse), interval=postplotinterval, out=None)
energy_r = ObservableReporter(IncompressibleKineticEnergy(lattice, flow_coarse), interval=postplotinterval, out=None)
spectrum_r = ObservableReporter(EnergySpectrum(lattice, flow_coarse), interval=postplotinterval, out=None)
print(f"                        Velocity    Density     Energy      Enstrophy")
for other_model in [learned_mrt, BGKCollision, RegularizedCollision, KBCCollision2D]:
    if other_model == MRTCollision:
        collision_coarse = other_model(lattice, D2Q9Dellar(lattice),[tau_coarse,tau_coarse,tau_coarse,tau_coarse,tau_coarse,tau_coarse,tau_coarse,tau_coarse,tau_coarse])
    elif other_model == learned_mrt:
        collision_coarse = other_model
    else:
        collision_coarse = other_model(lattice, tau_coarse)

    simulation_p = Simulation(flow=flow_coarse, lattice=lattice, collision=collision_coarse, streaming=streaming)
    simulation_p.flow.units.characteristic_velocity_pu = simulation.flow.units.characteristic_velocity_pu
    f_reporter_r = FReporter_coarse(data_interval)
    f_reporter_r.fs = []
    f_reporter_r.ts = []
    simulation_p.reporters.append(f_reporter_r)
    simulation_p.reporters.append(enstrophy_r)
    simulation_p.reporters.append(energy_r)
    simulation_p.reporters.append(spectrum_r)
    with torch.no_grad():
        simulation_p.step(n_steps-nr_init*data_interval)

    error_rho = (loss_function(lattice.rho(f_output.cuda()), lattice.rho(simulation_p.f)).item())
    error_u = (loss_function(lattice.u(f_output.cuda()), lattice.u(simulation_p.f)).item())
    error_energy = (loss_function(energy(f_output.cuda()), energy(simulation_p.f)).item())
    error_enstrophy = (loss_function(enstrophy(f_output.cuda()), enstrophy(simulation_p.f)).item())

    print(
        f"{other_model.__name__:<20s}    {np.mean(error_u):.4e}  {np.mean(error_rho):.4e}  {np.mean(error_energy):.4e}  {np.mean(error_enstrophy):.4e}")

    energy_data.append([[other_model.__name__], [energy_r.out]])
    enstrophy_data.append([[other_model.__name__], [enstrophy_r.out]])
    spectrum_data.append([[other_model.__name__], [np.array(spectrum_r.out)[-1, 2:]]])
    f_image.append([[other_model.__name__], [f_reporter_r.fs]])

    enstrophy_r.out = []
    energy_r.out = []
    spectrum_r.out = []
    energy_c_data = torch.zeros(len(data))
    enstrophy_c_data = torch.zeros(len(data))
    for i in range(len(data)):
        energy_c_data[i] = energy(data[i].cuda())
        enstrophy_c_data[i] = enstrophy(data[i].cuda())
    energy_c_data = lattice.convert_to_numpy(energy_c_data)
    enstrophy_c_data = lattice.convert_to_numpy(enstrophy_c_data)

########################################################################################################################
##### Plotting
########################################################################################################################
plt.rc('font', family='serif')
plt.rc('font', size=10)
plt.rcParams['text.latex.preview'] = True
rat = .5
columnWidth = 3.5
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.markeredgewidth'] = 1
plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['markers.fillstyle'] = 'full'

comparison = True
if comparison:
    def vorticity(f):
        u = lattice.convert_to_numpy(lattice.u(f))
        #dx = flow_coarse.units.convert_length_to_pu(1.0)
        grad_u0 = np.gradient(u[0], 1)
        grad_u1 = np.gradient(u[1], 1)
        vorticity = (grad_u1[0] - grad_u0[1])
        return vorticity

    def vorticity_plot(ax, f):
        u = lattice.convert_to_numpy(lattice.u(f))
        grad_u0 = np.gradient(u[0], 1)
        grad_u1 = np.gradient(u[1], 1)
        vorticity = (grad_u1[0] - grad_u0[1])
        image = ax.imshow(vorticity, cmap=cmap,vmin=min,vmax=max)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xticks([])
        ax.set_yticks([])
        return image

    fig, axes = plt.subplots(4, 4, figsize=(7, 7), sharex=True, sharey=True)
    nr = len(f_image[0][1][0]) - 1

    max = np.max(vorticity(data[int(nr*0.25)+nr_init].cuda()))
    min = np.min(vorticity(data[int(nr*0.25)+nr_init].cuda()))
    vorticity_plot(axes[0, 0], data[int(nr*0.25)+nr_init].cuda())
    vorticity_plot(axes[1, 0], f_image[0][1][0][int(nr*0.25)].cuda())
    vorticity_plot(axes[2, 0], f_image[1][1][0][int(nr*0.25)].cuda())
    vorticity_plot(axes[3, 0], f_image[2][1][0][int(nr*0.25)].cuda())

    max = np.max(vorticity(data[int(nr*0.5)+nr_init].cuda()))
    min = np.min(vorticity(data[int(nr*0.5)+nr_init].cuda()))
    vorticity_plot(axes[0, 1], data[int(nr*0.5)+nr_init].cuda())
    vorticity_plot(axes[1, 1], f_image[0][1][0][int(nr*0.5)].cuda())
    vorticity_plot(axes[2, 1], f_image[1][1][0][int(nr*0.5)].cuda())
    vorticity_plot(axes[3, 1], f_image[2][1][0][int(nr*0.5)].cuda())


    max = np.max(vorticity(data[int(nr*0.75)+nr_init].cuda()))
    min = np.min(vorticity(data[int(nr*0.75)+nr_init].cuda()))
    vorticity_plot(axes[0, 2], data[int(nr*0.75)+nr_init].cuda())
    vorticity_plot(axes[1, 2], f_image[0][1][0][int(nr*0.75)].cuda())
    vorticity_plot(axes[2, 2], f_image[1][1][0][int(nr*0.75)].cuda())
    vorticity_plot(axes[3, 2], f_image[2][1][0][int(nr*0.75)].cuda())

    max = np.max(vorticity(data[int(nr)+nr_init].cuda()))
    min = np.min(vorticity(data[int(nr)+nr_init].cuda()))
    vorticity_plot(axes[0, 3], data[int(nr)+nr_init].cuda())
    im1 = vorticity_plot(axes[1, 3], f_image[0][1][0][int(nr)].cuda())
    vorticity_plot(axes[2, 3], f_image[1][1][0][int(nr)].cuda())
    vorticity_plot(axes[3, 3], f_image[2][1][0][int(nr)].cuda())

    axes[0, 0].set_ylabel(f'BGK \n'+r' $128 \times 128$')
    axes[1, 0].set_ylabel('Neural LBM \n'+r' $64 \times 64$')
    axes[2, 0].set_ylabel('BGK \n'+r' $64 \times 64$')
    axes[3, 0].set_ylabel('Regularized \n'+r' $64 \times 64$')
    axes[0, 0].set_title(f'step= 1000', fontsize='medium')
    axes[0, 1].set_title(r'2000', fontsize='medium')
    axes[0, 2].set_title(r'3000', fontsize='medium')
    axes[0, 3].set_title(r'4000', fontsize='medium')
    plt.tight_layout(pad=0.2, w_pad=0.2)
    outputFileName = "./vorticity"
    #plt.savefig(outputFileName + ".pdf", format='pdf', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.savefig(outputFileName + ".png", format='png', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()
