#!/usr/bin/env python3
"""Generate visualization for README.md"""

import numpy as np
import matplotlib.pyplot as plt

def generate_visualization():
    """Generate a sample visualization for the README"""
    # Create mock data for visualization
    s_grid = np.linspace(0, 200, 50)
    t_grid = np.linspace(0, 1, 50)
    
    # Create mock option prices (Black-Scholes-like surface)
    S, T = np.meshgrid(s_grid, t_grid)
    # Simple Black-Scholes-like payoff (this is just for visualization)
    prices = np.maximum(S - 100, 0) * np.exp(-0.05 * (1 - T))
    
    # Create mock Greeks
    # Delta is approximately 1 for S >> K and 0 for S << K
    delta = np.where(S > 100, 1 - np.exp(-0.05 * (1 - T)), 0)
    # Gamma is highest near the strike
    gamma = np.exp(-((S - 100)**2) / (2 * 50**2)) * (1 - T)
    # Theta is negative (option value decreases with time)
    theta = -np.maximum(S - 100, 0) * 0.05 * np.exp(-0.05 * (1 - T))
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(12, 10))
    
    # Plot 1: Price surface
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf = ax1.plot_surface(S, T, prices, cmap='viridis', 
                           linewidth=0, antialiased=True)
    ax1.set_xlabel('Asset Price')
    ax1.set_ylabel('Time')
    ax1.set_zlabel('Option Price')
    ax1.set_title('Option Price Surface')
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    # Plot 2: Price heatmap
    ax2 = fig.add_subplot(2, 2, 2)
    im2 = ax2.imshow(prices, aspect='auto', cmap='viridis', 
                     extent=[0, 1, 0, 200], origin='lower')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Asset Price')
    ax2.set_title('Option Price Heatmap')
    fig.colorbar(im2, ax=ax2)
    
    # Plot 3: Delta heatmap
    ax3 = fig.add_subplot(2, 2, 3)
    im3 = ax3.imshow(delta, aspect='auto', cmap='RdBu', 
                     extent=[0, 1, 0, 200], origin='lower')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Asset Price')
    ax3.set_title('Delta Heatmap')
    fig.colorbar(im3, ax=ax3)
    
    # Plot 4: Gamma heatmap
    ax4 = fig.add_subplot(2, 2, 4)
    im4 = ax4.imshow(gamma, aspect='auto', cmap='RdBu', 
                     extent=[0, 1, 0, 200], origin='lower')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Asset Price')
    ax4.set_title('Gamma Heatmap')
    fig.colorbar(im4, ax=ax4)
    
    plt.tight_layout()
    plt.savefig('pde_solution_example.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualization saved as pde_solution_example.png")

if __name__ == "__main__":
    generate_visualization()