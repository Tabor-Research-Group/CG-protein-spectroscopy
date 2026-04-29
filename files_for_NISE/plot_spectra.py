import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# Load data
Data_predicted = np.loadtxt('Absorption_pred.dat')
Data_original = np.loadtxt('Absorption_gt.dat')

# Normalize intensities to [0, 1]
# Each spectrum is normalized independently to its own maximum
Data_predicted_norm = Data_predicted.copy()
Data_predicted_norm[:, 1] = Data_predicted[:, 1] / np.max(Data_predicted[:, 1])

Data_original_norm = Data_original.copy()
Data_original_norm[:, 1] = Data_original[:, 1] / np.max(Data_original[:, 1])

# Plot normalized data
plt.plot(Data_predicted_norm[:, 0], Data_predicted_norm[:, 1], 
         label='Predicted Hamiltonians +\nPredicted Dipoles', 
         color='red', linewidth=2)
plt.plot(Data_original_norm[:, 0], Data_original_norm[:, 1], 
         label='Electrostatic Maps+\nAtomistic Dipole', 
         color='blue', linewidth=2)

# Labeling
plt.xlabel('$\omega$ [cm$^{-1}$]', fontsize=16)
plt.ylabel('Normalized Absorption [arb.u.]', fontsize=16)

# Legend with better placement
plt.legend(fontsize=12, frameon=False, loc='best')

# Set y-axis limits for normalized data
plt.ylim(0, 1.05)

# Save figure
plt.tight_layout()
plt.savefig('2fyg_A_plot.png', dpi=600, bbox_inches='tight')
plt.close()
