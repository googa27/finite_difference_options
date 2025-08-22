"""Heston Model Demonstration.

This script demonstrates the multi-dimensional finite difference pricing framework
using the Heston stochastic volatility model for option pricing.

Features:
- European call and put option pricing
- Volatility smile analysis
- Correlation impact studies
- Greeks computation
- Surface plotting
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import with absolute paths to avoid relative import issues
sys.path.append(str(Path(__file__).parent.parent))
from src.multidimensional_processes import HestonModel
from src.multidimensional_solver import create_adi_solver
from src.multidimensional_pricing_engine import (
    MultiDimensionalPricingEngine,
    create_european_call_2d,
    create_european_put_2d,
    create_log_price_grid,
    create_variance_grid,
)
from src.multidimensional_boundary_conditions import create_heston_boundaries


def setup_heston_parameters():
    """Set up standard Heston model parameters."""
    return {
        'initial_price': 100.0,
        'initial_variance': 0.04,  # 20% volatility
        'risk_free_rate': 0.05,
        'dividend_yield': 0.0,
        'kappa': 2.0,  # Mean reversion speed
        'theta': 0.04,  # Long-term variance
        'sigma': 0.3,  # Volatility of volatility
        'rho': -0.7,  # Correlation between price and volatility
    }


def create_heston_setup(params, strike=100.0, maturity=0.25, nx=51, nv=31, nt=50):
    """Create Heston model setup for pricing."""
    # Create Heston process
    process = HestonModel(
        kappa=params['kappa'],
        theta=params['theta'],
        sigma=params['sigma'],
        rho=params['rho'],
        risk_free_rate=params['risk_free_rate'],
        dividend_yield=params['dividend_yield']
    )
    
    # Create grids
    s_min, s_max = 50.0, 150.0
    v_min, v_max = 0.001, 1.0
    
    s_grid = create_log_price_grid(s_min, s_max, nx, center=params['initial_price'])
    v_grid = create_variance_grid(v_min, v_max, nv)
    t_grid = np.linspace(0, maturity, nt)
    
    # Create solver
    solver = create_adi_solver(theta=0.5)  # Crank-Nicolson
    
    # Create pricing engine
    engine = MultiDimensionalPricingEngine(process, solver)
    
    return engine, s_grid, v_grid, t_grid


def price_european_options():
    """Price European call and put options using Heston model."""
    print("=== European Option Pricing with Heston Model ===")
    
    params = setup_heston_parameters()
    strike = 100.0
    maturity = 0.25
    
    engine, s_grid, v_grid, t_grid = create_heston_setup(params, strike, maturity)
    
    # Create options
    call_option = create_european_call_2d(strike, maturity)
    put_option = create_european_put_2d(strike, maturity)
    
    # Create boundary conditions
    call_boundaries = create_heston_boundaries(
        s_grid[0], s_grid[-1], v_grid[-1], strike, is_call=True
    )
    put_boundaries = create_heston_boundaries(
        s_grid[0], s_grid[-1], v_grid[-1], strike, is_call=False
    )
    
    print(f"Grid sizes: S={len(s_grid)}, V={len(v_grid)}, T={len(t_grid)}")
    print(f"Strike: {strike}, Maturity: {maturity}")
    print(f"Initial conditions: S0={params['initial_price']}, V0={params['initial_variance']}")
    
    # Price call option
    start_time = time.time()
    call_prices = engine.price_option(call_option, s_grid, v_grid, t_grid, call_boundaries)
    call_time = time.time() - start_time
    
    # Price put option
    start_time = time.time()
    put_prices = engine.price_option(put_option, s_grid, v_grid, t_grid, put_boundaries)
    put_time = time.time() - start_time
    
    # Extract prices at initial conditions
    s_idx = np.argmin(np.abs(s_grid - params['initial_price']))
    v_idx = np.argmin(np.abs(v_grid - params['initial_variance']))
    
    call_price = call_prices[-1, s_idx, v_idx]  # At maturity
    put_price = put_prices[-1, s_idx, v_idx]
    
    print(f"\nResults:")
    print(f"Call option price: {call_price:.4f} (computed in {call_time:.3f}s)")
    print(f"Put option price: {put_price:.4f} (computed in {put_time:.3f}s)")
    print(f"Put-call parity check: C - P = {call_price - put_price:.4f}, "
          f"S - K*exp(-rT) = {params['initial_price'] - strike * np.exp(-params['risk_free_rate'] * maturity):.4f}")
    
    return call_prices, put_prices, s_grid, v_grid, t_grid


def analyze_volatility_smile():
    """Analyze volatility smile for different strikes."""
    print("\n=== Volatility Smile Analysis ===")
    
    params = setup_heston_parameters()
    maturity = 0.25
    strikes = np.linspace(80, 120, 9)
    
    call_prices = []
    
    for strike in strikes:
        engine, s_grid, v_grid, t_grid = create_heston_setup(
            params, strike, maturity, nx=41, nv=21, nt=30
        )
        
        call_option = create_european_call_2d(strike, maturity)
        boundaries = create_heston_boundaries(
            s_grid[0], s_grid[-1], v_grid[-1], strike, is_call=True
        )
        
        prices = engine.price_option(call_option, s_grid, v_grid, t_grid, boundaries)
        
        # Extract price at initial conditions
        s_idx = np.argmin(np.abs(s_grid - params['initial_price']))
        v_idx = np.argmin(np.abs(v_grid - params['initial_variance']))
        call_prices.append(prices[-1, s_idx, v_idx])
    
    # Plot volatility smile
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, call_prices, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Strike Price')
    plt.ylabel('Call Option Price')
    plt.title('Heston Model: Call Option Prices vs Strike')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('heston_volatility_smile.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Strike prices: {strikes}")
    print(f"Call prices: {np.array(call_prices)}")
    
    return strikes, call_prices


def correlation_impact_study():
    """Study the impact of correlation on option prices."""
    print("\n=== Correlation Impact Study ===")
    
    params = setup_heston_parameters()
    correlations = np.linspace(-0.9, 0.5, 8)
    strike = 100.0
    maturity = 0.25
    
    call_prices = []
    
    for rho in correlations:
        # Update correlation
        params_copy = params.copy()
        params_copy['rho'] = rho
        
        engine, s_grid, v_grid, t_grid = create_heston_setup(
            params_copy, strike, maturity, nx=41, nv=21, nt=30
        )
        
        call_option = create_european_call_2d(strike, maturity)
        boundaries = create_heston_boundaries(
            s_grid[0], s_grid[-1], v_grid[-1], strike, is_call=True
        )
        
        prices = engine.price_option(call_option, s_grid, v_grid, t_grid, boundaries)
        
        # Extract price at initial conditions
        s_idx = np.argmin(np.abs(s_grid - params['initial_price']))
        v_idx = np.argmin(np.abs(v_grid - params['initial_variance']))
        call_prices.append(prices[-1, s_idx, v_idx])
    
    # Plot correlation impact
    plt.figure(figsize=(10, 6))
    plt.plot(correlations, call_prices, 'ro-', linewidth=2, markersize=6)
    plt.xlabel('Correlation (ρ)')
    plt.ylabel('Call Option Price')
    plt.title('Heston Model: Impact of Correlation on Call Option Price')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('heston_correlation_impact.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Correlations: {correlations}")
    print(f"Call prices: {np.array(call_prices)}")
    
    return correlations, call_prices


def compute_greeks():
    """Compute and display Greeks for Heston model."""
    print("\n=== Greeks Computation ===")
    
    params = setup_heston_parameters()
    strike = 100.0
    maturity = 0.25
    
    engine, s_grid, v_grid, t_grid = create_heston_setup(
        params, strike, maturity, nx=51, nv=31, nt=50
    )
    
    call_option = create_european_call_2d(strike, maturity)
    boundaries = create_heston_boundaries(
        s_grid[0], s_grid[-1], v_grid[-1], strike, is_call=True
    )
    
    # Price option
    prices = engine.price_option(call_option, s_grid, v_grid, t_grid, boundaries)
    
    # Compute Greeks
    greeks = engine.compute_greeks(prices, s_grid, v_grid, t_grid)
    
    # Extract Greeks at initial conditions
    s_idx = np.argmin(np.abs(s_grid - params['initial_price']))
    v_idx = np.argmin(np.abs(v_grid - params['initial_variance']))
    t_idx = -1  # At maturity
    
    delta = greeks['delta'][t_idx, s_idx, v_idx]
    gamma = greeks['gamma'][t_idx, s_idx, v_idx]
    vega = greeks['vega'][t_idx, s_idx, v_idx]
    theta = greeks['theta'][t_idx, s_idx, v_idx]
    
    print(f"Greeks at S={params['initial_price']}, V={params['initial_variance']}:")
    print(f"Delta: {delta:.4f}")
    print(f"Gamma: {gamma:.6f}")
    print(f"Vega: {vega:.4f}")
    print(f"Theta: {theta:.4f}")
    
    return greeks


def plot_price_surface():
    """Plot 3D price surface."""
    print("\n=== Price Surface Visualization ===")
    
    params = setup_heston_parameters()
    strike = 100.0
    maturity = 0.25
    
    engine, s_grid, v_grid, t_grid = create_heston_setup(
        params, strike, maturity, nx=31, nv=21, nt=30
    )
    
    call_option = create_european_call_2d(strike, maturity)
    boundaries = create_heston_boundaries(
        s_grid[0], s_grid[-1], v_grid[-1], strike, is_call=True
    )
    
    prices = engine.price_option(call_option, s_grid, v_grid, t_grid, boundaries)
    
    # Create meshgrid for plotting
    S, V = np.meshgrid(s_grid, v_grid, indexing='ij')
    
    # Plot 3D surface
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surface = ax.plot_surface(S, V, prices[-1], cmap='viridis', alpha=0.8)
    ax.set_xlabel('Spot Price (S)')
    ax.set_ylabel('Variance (V)')
    ax.set_zlabel('Option Price')
    ax.set_title('Heston Model: Call Option Price Surface')
    
    plt.colorbar(surface, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig('heston_price_surface.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot 2D contour
    plt.figure(figsize=(10, 6))
    contour = plt.contour(S, V, prices[-1], levels=15)
    plt.clabel(contour, inline=True, fontsize=8)
    plt.xlabel('Spot Price (S)')
    plt.ylabel('Variance (V)')
    plt.title('Heston Model: Call Option Price Contours')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('heston_price_contours.png', dpi=150, bbox_inches='tight')
    plt.show()


def performance_benchmark():
    """Benchmark performance for different grid sizes."""
    print("\n=== Performance Benchmark ===")
    
    params = setup_heston_parameters()
    strike = 100.0
    maturity = 0.25
    
    grid_sizes = [(21, 11, 20), (31, 21, 30), (41, 31, 40), (51, 41, 50)]
    times = []
    
    for nx, nv, nt in grid_sizes:
        engine, s_grid, v_grid, t_grid = create_heston_setup(
            params, strike, maturity, nx, nv, nt
        )
        
        call_option = create_european_call_2d(strike, maturity)
        boundaries = create_heston_boundaries(
            s_grid[0], s_grid[-1], v_grid[-1], strike, is_call=True
        )
        
        start_time = time.time()
        prices = engine.price_option(call_option, s_grid, v_grid, t_grid, boundaries)
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        total_points = nx * nv * nt
        print(f"Grid {nx}×{nv}×{nt} ({total_points:,} points): {elapsed:.3f}s")
    
    # Plot performance
    total_points = [nx * nv * nt for nx, nv, nt in grid_sizes]
    
    plt.figure(figsize=(10, 6))
    plt.loglog(total_points, times, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Total Grid Points')
    plt.ylabel('Computation Time (seconds)')
    plt.title('Heston Model: Performance Scaling')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('heston_performance.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return grid_sizes, times


def main():
    """Run complete Heston model demonstration."""
    print("Heston Model Multi-Dimensional Pricing Framework Demo")
    print("=" * 55)
    
    # Create output directory for plots
    Path("heston_plots").mkdir(exist_ok=True)
    
    try:
        # 1. Basic option pricing
        call_prices, put_prices, s_grid, v_grid, t_grid = price_european_options()
        
        # 2. Volatility smile analysis
        strikes, smile_prices = analyze_volatility_smile()
        
        # 3. Correlation impact study
        correlations, corr_prices = correlation_impact_study()
        
        # 4. Greeks computation
        greeks = compute_greeks()
        
        # 5. Price surface visualization
        plot_price_surface()
        
        # 6. Performance benchmark
        grid_sizes, times = performance_benchmark()
        
        print("\n" + "=" * 55)
        print("Demo completed successfully!")
        print("Generated plots:")
        print("- heston_volatility_smile.png")
        print("- heston_correlation_impact.png") 
        print("- heston_price_surface.png")
        print("- heston_price_contours.png")
        print("- heston_performance.png")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
