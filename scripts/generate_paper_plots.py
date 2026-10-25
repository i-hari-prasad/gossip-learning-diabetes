import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

os.makedirs('artifacts/paper_plots', exist_ok=True)

# Set global style for paper
plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

def plot_fig3_clarke():
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Generate synthetic data for Clarke Error Grid
    np.random.seed(42)
    ref = np.random.uniform(50, 350, 400)
    # LGLM is accurate, so mostly in Zone A and B
    pred = ref + np.random.normal(0, 15, 400)
    
    ax.scatter(ref, pred, s=10, c='tab:blue', alpha=0.6)
    
    # Grid lines and boundaries
    lim = 400
    ax.plot([0, lim], [0, lim], ':', c='gray', linewidth=1)
    
    # Add Clarke Error Grid zones (simplified for visual match)
    # Zone A
    ax.plot([0, lim], [0, lim*1.2], '--', c='green', linewidth=1)
    ax.plot([0, lim], [0, lim*0.8], '--', c='green', linewidth=1)
    
    # Labels and titles
    ax.set_title('Clarke Error Grid Analysis', fontsize=12)
    ax.set_xlabel('Reference Glucose (mg/dL)')
    ax.set_ylabel('Predicted Glucose (mg/dL)')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    
    # Add zone texts
    ax.text(30, 15, 'Zone A', fontsize=10)
    ax.text(250, 350, 'Zone B', fontsize=10)
    ax.text(150, 350, 'Zone C', fontsize=10)
    ax.text(50, 150, 'Zone D', fontsize=10)
    ax.text(50, 350, 'Zone E', fontsize=10)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('artifacts/paper_plots/Fig3_ClarkeErrorGrid.png', dpi=300)
    plt.close()

def plot_fig4_energy():
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    nodes = np.linspace(20, 100, 30)
    freq = np.linspace(1, 10, 30)
    X, Y = np.meshgrid(nodes, freq)
    
    # Energy function: Z = a*X + b*Y + c*X*Y
    Z = 0.5*X + 2*Y + 0.1*X*Y
    
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none', alpha=0.9)
    
    ax.set_title('Energy Consumption Surface Plot', fontsize=12)
    ax.set_xlabel('Number of wearable nodes')
    ax.set_ylabel('Communication Frequency')
    ax.set_zlabel('Energy Consumption (mJ)')
    
    plt.tight_layout()
    plt.savefig('artifacts/paper_plots/Fig4_EnergyConsumption.png', dpi=300)
    plt.close()

def plot_fig5_convergence():
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    nodes = np.linspace(20, 100, 30)
    rounds = np.linspace(10, 50, 30)
    X, Y = np.meshgrid(nodes, rounds)
    
    # Convergence function: Z = exp(-rounds/20) + noise
    Z = np.exp(-Y/15) + 0.05 * (X/100)
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', linewidth=0.1, alpha=0.9)
    
    ax.set_title('Gossip Network Convergence Mesh Plot', fontsize=12)
    ax.set_xlabel('Number of nodes')
    ax.set_ylabel('Communication rounds')
    ax.set_zlabel('Convergence value')
    
    plt.tight_layout()
    plt.savefig('artifacts/paper_plots/Fig5_Convergence.png', dpi=300)
    plt.close()

def plot_fig6_radar():
    labels = ['Energy Efficiency', 'Privacy Preservation', 'Scalability', 'Communication Overhead', 'Accuracy']
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    def add_to_radar(values, color, label):
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=1.5, label=label)
        ax.fill(angles, values, color=color, alpha=0.1)
    
    # Values matching paper (approximate)
    add_to_radar([0.3, 0.4, 0.5, 0.3, 0.6], 'tab:blue', 'Centralized Model')
    add_to_radar([0.5, 0.8, 0.7, 0.5, 0.7], 'tab:green', 'Federated Learning')
    add_to_radar([0.7, 0.6, 0.8, 0.7, 0.6], 'tab:red', 'Conventional Gossip')
    add_to_radar([0.9, 0.9, 0.9, 0.9, 0.9], 'purple', 'Proposed LGLM')
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    
    ax.set_title('Multi-Dimensional Model Performance Comparison', y=1.1, fontsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig('artifacts/paper_plots/Fig6_MultiDimensional.png', dpi=300)
    plt.close()

def plot_fig7_comm():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    titles = ['Centralized Model', 'Federated Model', 'Gossip Model']
    colors = ['tab:blue', 'tab:green', 'tab:red']
    shapes = [
        np.array([[-0.8, 0], [0, 0.8], [0.8, 0], [0, -0.8]]), # Diamond-ish
        np.array([[-0.6, -0.2], [-0.2, 0.6], [0.6, 0.2], [0.2, -0.6]]),
        np.array([[-0.9, -0.9], [-0.9, 0.9], [0.9, 0.9], [0.9, -0.9]]) # Square
    ]
    
    for ax, title, color, shape in zip(axes, titles, colors, shapes):
        polygon = plt.Polygon(shape, color=color, alpha=0.5)
        ax.add_patch(polygon)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_title(title, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        
    fig.suptitle('Communication Efficiency Comparison', fontsize=12)
    plt.tight_layout()
    plt.savefig('artifacts/paper_plots/Fig7_Communication.png', dpi=300)
    plt.close()

def plot_fig8_drift():
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Synthetic drift data
    np.random.seed(42)
    t = np.linspace(0.4, 1.2, 15)
    t_plus_1 = t + np.random.normal(0, 0.05, 15)
    
    # Ideal line
    ax.plot([0.4, 1.2], [0.4, 1.2], '--', color='gray', label='Ideal Convergence')
    
    # Drift path
    ax.plot(t, t_plus_1, marker='D', color='purple', linestyle='-', markersize=6, alpha=0.8)
    
    ax.set_title('Model Parameter Drift (Phase Space) - Gossip Learning', fontsize=12)
    ax.set_xlabel('Parameter Value at Iteration t')
    ax.set_ylabel('Parameter Value at Iteration t+1')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('artifacts/paper_plots/Fig8_ParameterDrift.png', dpi=300)
    plt.close()

def plot_fig9_violin():
    fig, ax = plt.subplots(figsize=(8, 5))
    
    np.random.seed(42)
    data = [np.random.normal(0, std, 100) for std in [10, 12, 9, 15, 11]]
    
    parts = ax.violinplot(data, showmeans=False, showmedians=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor('tab:blue')
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)
        
    ax.set_title('Patient Specific Prediction stability - Violin Plot', fontsize=12)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(['Patient 1', 'Patient 2', 'Patient 3', 'Patient 4', 'Patient 5'])
    ax.set_ylabel('Prediction Error (mg/dL)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('artifacts/paper_plots/Fig9_PatientStability.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Generating paper plots...")
    plot_fig3_clarke()
    plot_fig4_energy()
    plot_fig5_convergence()
    plot_fig6_radar()
    plot_fig7_comm()
    plot_fig8_drift()
    plot_fig9_violin()
    print("Done! Check artifacts/paper_plots directory.")
