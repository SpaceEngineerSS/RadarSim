#!/usr/bin/env python3
"""
Performance Plotting Script

Generates publication-quality Pd (Probability of Detection) curves
from batch simulation results.

Usage:
    python plot_performance.py output/batch_results.csv
    python plot_performance.py output/batch_results.csv --save figures/

Requires:
    pip install matplotlib pandas seaborn
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Plotting requires: pip install pandas matplotlib")


def load_results(filepath: str) -> 'pd.DataFrame':
    """Load simulation results from CSV."""
    return pd.read_csv(filepath)


def plot_pd_vs_range(df: 'pd.DataFrame', save_path: Optional[str] = None) -> None:
    """
    Plot Probability of Detection vs Range for different RCS values.
    
    This is the classic radar performance curve.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by RCS and range, take mean of detection ratio
    rcs_values = sorted(df['rcs_m2'].unique())
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(rcs_values)))
    
    for rcs, color in zip(rcs_values, colors):
        subset = df[df['rcs_m2'] == rcs]
        grouped = subset.groupby('range_km')['detection_ratio'].agg(['mean', 'std'])
        
        ax.plot(
            grouped.index, grouped['mean'],
            'o-', color=color, label=f'σ = {rcs} m²',
            linewidth=2, markersize=6
        )
        
        # Error bands
        ax.fill_between(
            grouped.index,
            grouped['mean'] - grouped['std'],
            grouped['mean'] + grouped['std'],
            alpha=0.2, color=color
        )
    
    ax.set_xlabel('Range (km)', fontsize=12)
    ax.set_ylabel('Probability of Detection ($P_d$)', fontsize=12)
    ax.set_title('Detection Probability vs Range\n(Various RCS Values)', fontsize=14)
    ax.legend(title='Target RCS', loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, df['range_km'].max() * 1.05)
    
    plt.tight_layout()
    
    if save_path:
        filepath = os.path.join(save_path, 'pd_vs_range.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
    else:
        plt.show()


def plot_pd_vs_snr(df: 'pd.DataFrame', save_path: Optional[str] = None) -> None:
    """
    Plot Probability of Detection vs SNR.
    
    Shows system noise tolerance.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bin SNR values
    df['snr_bin'] = pd.cut(df['mean_snr_db'], bins=20)
    grouped = df.groupby('snr_bin')['detection_ratio'].mean()
    
    # Extract bin centers
    bin_centers = [interval.mid for interval in grouped.index]
    
    ax.plot(bin_centers, grouped.values, 'o-', color='#00d4ff', 
            linewidth=2, markersize=8)
    
    # Add threshold line
    threshold = df['threshold_db'].iloc[0] if 'threshold_db' in df.columns else 13
    ax.axvline(threshold, color='red', linestyle='--', 
               label=f'Threshold = {threshold} dB')
    
    ax.set_xlabel('Signal-to-Noise Ratio (dB)', fontsize=12)
    ax.set_ylabel('Probability of Detection ($P_d$)', fontsize=12)
    ax.set_title('Detection Probability vs SNR', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        filepath = os.path.join(save_path, 'pd_vs_snr.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
    else:
        plt.show()


def plot_snr_heatmap(df: 'pd.DataFrame', save_path: Optional[str] = None) -> None:
    """
    Plot SNR heatmap across range and RCS.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Pivot table
    pivot = df.pivot_table(
        values='mean_snr_db',
        index='rcs_m2',
        columns='range_km',
        aggfunc='mean'
    )
    
    im = ax.imshow(pivot.values, aspect='auto', cmap='plasma',
                   extent=[pivot.columns.min(), pivot.columns.max(),
                          pivot.index.min(), pivot.index.max()],
                   origin='lower')
    
    plt.colorbar(im, ax=ax, label='Mean SNR (dB)')
    
    ax.set_xlabel('Range (km)', fontsize=12)
    ax.set_ylabel('RCS (m²)', fontsize=12)
    ax.set_title('SNR Heatmap: Range vs RCS', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        filepath = os.path.join(save_path, 'snr_heatmap.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
    else:
        plt.show()


def generate_report(df: 'pd.DataFrame', save_path: str) -> None:
    """Generate all plots and summary statistics."""
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n{'=' * 60}")
    print("PERFORMANCE ANALYSIS REPORT")
    print(f"{'=' * 60}")
    
    # Summary statistics
    print(f"\nTotal simulations: {len(df)}")
    print(f"Range: {df['range_km'].min():.0f} - {df['range_km'].max():.0f} km")
    print(f"RCS values: {sorted(df['rcs_m2'].unique())}")
    print(f"Overall Pd: {df['detection_ratio'].mean():.3f}")
    print(f"Mean SNR: {df['mean_snr_db'].mean():.1f} dB")
    
    # Generate plots
    print(f"\nGenerating plots...")
    plot_pd_vs_range(df, save_path)
    plot_pd_vs_snr(df, save_path)
    plot_snr_heatmap(df, save_path)
    
    # Save summary CSV
    summary = df.groupby(['range_km', 'rcs_m2']).agg({
        'detection_ratio': ['mean', 'std'],
        'mean_snr_db': 'mean',
        'n_pulses': 'sum',
        'n_detections': 'sum'
    }).round(3)
    
    summary_path = os.path.join(save_path, 'summary_stats.csv')
    summary.to_csv(summary_path)
    print(f"✓ Saved: {summary_path}")
    
    print(f"\n{'=' * 60}")
    print(f"Report complete: {save_path}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot radar performance from batch results"
    )
    parser.add_argument(
        "input", type=str,
        help="Input CSV file from batch_run.py"
    )
    parser.add_argument(
        "--save", type=str, default="output/figures",
        help="Directory to save figures (default: output/figures)"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show plots interactively instead of saving"
    )
    
    args = parser.parse_args()
    
    if not PLOTTING_AVAILABLE:
        print("Error: Plotting dependencies not available")
        print("Install with: pip install pandas matplotlib")
        return 1
    
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        return 1
    
    # Load data
    df = load_results(args.input)
    print(f"Loaded {len(df)} results from: {args.input}")
    
    # Generate report
    save_path = None if args.show else args.save
    generate_report(df, save_path or "output/figures")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
