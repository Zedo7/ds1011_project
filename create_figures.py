"""
Publication-Quality Figure Generation
======================================

Creates aesthetic, clear, and novel visualizations for:
- Conference poster presentation
- Final report/paper
- Supplementary materials

Figures include:
1. Main MASE comparison (bar + error bars)
2. LES extrapolation curves
3. Multi-dataset heatmap
4. Statistical significance visualization
5. RoPE vs xPos detailed comparison
6. ALiBi failure mode analysis

Author: DS1011 Final Project
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

# Publication-quality settings
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.axisbelow': True,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Color palettes
PALETTE_MAIN = {
    'rope': '#2E7D32',      # Deep green (best)
    'xpos': '#388E3C',      # Light green (also good)
    'absolute': '#F57C00', # Orange (moderate)
    'alibi': '#D32F2F',    # Red (worst)
    'none': '#757575'       # Gray (baseline)
}

PALETTE_DATASET = {
    'electricity': '#1976D2',  # Blue
    'weather': '#388E3C',      # Green
    'traffic': '#D32F2F'       # Red
}

def load_data():
    """Load experimental results and statistics."""
    print("Loading data...")
    df = pd.read_csv('reports/tables/extrapolation.csv')
    
    try:
        stats_df = pd.read_csv('reports/tables/comprehensive_statistics.csv')
    except FileNotFoundError:
        print("⚠ Statistics not found. Run analyze_multiseed.py first.")
        stats_df = None
    
    print(f"✓ Loaded {len(df)} evaluations across {df['dataset'].nunique()} datasets")
    return df, stats_df

def figure1_mase_comparison(df, stats_df):
    """
    Figure 1: Main MASE comparison across datasets
    Professional bar chart with error bars, color-coded by performance
    """
    print("\nGenerating Figure 1: MASE Comparison...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    datasets = sorted(df['dataset'].unique())
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        data = df[df['dataset'] == dataset]
        
        # Compute statistics
        pe_stats = []
        for pe in sorted(data['pe'].unique()):
            pe_data = data[data['pe'] == pe]['MASE'].values
            mean = np.mean(pe_data)
            sem = np.std(pe_data, ddof=1) / np.sqrt(len(pe_data)) if len(pe_data) > 1 else 0
            ci_95 = 1.96 * sem
            pe_stats.append((pe, mean, ci_95))
        
        pe_stats.sort(key=lambda x: x[1])  # Sort by mean
        
        pes = [x[0] for x in pe_stats]
        means = [x[1] for x in pe_stats]
        cis = [x[2] for x in pe_stats]
        colors = [PALETTE_MAIN[pe] for pe in pes]
        
        # Bar plot
        bars = ax.bar(range(len(pes)), means, yerr=cis, capsize=5,
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1.2,
                      error_kw={'linewidth': 1.5, 'ecolor': 'black'})
        
        # Add value labels
        for i, (mean, ci) in enumerate(zip(means, cis)):
            ax.text(i, mean + ci + 0.03, f'{mean:.2f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Baseline reference
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, 
                  alpha=0.6, label='Seasonal Naive (MASE=1.0)')
        
        # Styling
        ax.set_xticks(range(len(pes)))
        ax.set_xticklabels([pe.upper() for pe in pes], rotation=0)
        ax.set_ylabel('MASE (lower is better)', fontweight='bold')
        ax.set_title(f'{dataset.capitalize()}', fontweight='bold', fontsize=13)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_ylim([0, max(means) * 1.25])
        
        # Performance regions
        ax.axhspan(0, 1.0, alpha=0.1, color='green', label='Better than naive')
        ax.axhspan(1.0, ax.get_ylim()[1], alpha=0.1, color='red')
    
    plt.suptitle('Forecasting Accuracy Across Datasets and Positional Encodings',
                fontweight='bold', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig('reports/figures/fig1_mase_comparison.png')
    plt.savefig('reports/figures/fig1_mase_comparison.pdf')
    print("✓ Saved: fig1_mase_comparison.png")
    plt.close()

def figure2_les_curves(df):
    """
    Figure 2: Length Extrapolation Score curves
    Shows how performance degrades as context increases
    """
    print("\nGenerating Figure 2: LES Curves...")
    
    datasets = sorted(df['dataset'].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 4.5))
    
    if len(datasets) == 1:
        axes = [axes]
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        data = df[df['dataset'] == dataset]
        
        # Get LES data (only where L_eval > L_train)
        les_data = data[data['L_eval'] > data['L_train']]
        
        for pe in sorted(data['pe'].unique()):
            pe_data = les_data[les_data['pe'] == pe].groupby('L_eval')['LES'].agg(['mean', 'std'])
            
            if len(pe_data) > 0:
                ax.plot(pe_data.index, pe_data['mean'], marker='o', 
                       linewidth=2.5, markersize=10, label=pe.upper(),
                       color=PALETTE_MAIN[pe], alpha=0.9)
                
                # Add confidence interval if multiple seeds
                if pe_data['std'].max() > 0:
                    ax.fill_between(pe_data.index, 
                                   pe_data['mean'] - 1.96*pe_data['std'],
                                   pe_data['mean'] + 1.96*pe_data['std'],
                                   alpha=0.2, color=PALETTE_MAIN[pe])
        
        # Perfect extrapolation reference
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, 
                  alpha=0.6, label='Perfect (LES=1.0)')
        
        # Acceptable region
        ax.axhspan(0.95, 1.05, alpha=0.1, color='green')
        
        ax.set_xlabel('Evaluation Context Length', fontweight='bold')
        ax.set_ylabel('LES (closer to 1.0 is better)', fontweight='bold')
        ax.set_title(f'{dataset.capitalize()}', fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Length Extrapolation Performance',
                fontweight='bold', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig('reports/figures/fig2_les_curves.png')
    plt.savefig('reports/figures/fig2_les_curves.pdf')
    print("✓ Saved: fig2_les_curves.png")
    plt.close()

def figure3_multidataset_heatmap(df):
    """
    Figure 3: Multi-dataset performance heatmap
    Comprehensive view of all results
    """
    print("\nGenerating Figure 3: Multi-Dataset Heatmap...")
    
    # Create pivot table: (dataset, PE) × metrics
    pivot_mase = df.groupby(['dataset', 'pe'])['MASE'].mean().unstack()
    pivot_les = df[df['L_eval'] > df['L_train']].groupby(['dataset', 'pe'])['LES'].mean().unstack()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # MASE heatmap
    sns.heatmap(pivot_mase, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                center=1.0, vmin=0.5, vmax=1.5, cbar_kws={'label': 'MASE'},
                ax=ax1, linewidths=0.5, linecolor='gray')
    ax1.set_title('MASE by Dataset and PE\n(lower is better)', 
                 fontweight='bold', fontsize=13)
    ax1.set_xlabel('Positional Encoding', fontweight='bold')
    ax1.set_ylabel('Dataset', fontweight='bold')
    ax1.set_xticklabels([x.get_text().upper() for x in ax1.get_xticklabels()])
    ax1.set_yticklabels([x.get_text().capitalize() for x in ax1.get_yticklabels()], 
                       rotation=0)
    
    # LES heatmap
    sns.heatmap(pivot_les, annot=True, fmt='.4f', cmap='RdYlGn_r',
                center=1.0, vmin=0.99, vmax=1.03, cbar_kws={'label': 'LES'},
                ax=ax2, linewidths=0.5, linecolor='gray')
    ax2.set_title('LES by Dataset and PE\n(closer to 1.0 is better)',
                 fontweight='bold', fontsize=13)
    ax2.set_xlabel('Positional Encoding', fontweight='bold')
    ax2.set_ylabel('Dataset', fontweight='bold')
    ax2.set_xticklabels([x.get_text().upper() for x in ax2.get_xticklabels()])
    ax2.set_yticklabels([x.get_text().capitalize() for x in ax2.get_yticklabels()],
                       rotation=0)
    
    plt.suptitle('Comprehensive Performance Heatmap',
                fontweight='bold', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig('reports/figures/fig3_multidataset_heatmap.png')
    plt.savefig('reports/figures/fig3_multidataset_heatmap.pdf')
    print("✓ Saved: fig3_multidataset_heatmap.png")
    plt.close()

def figure4_rope_vs_xpos(df):
    """
    Figure 4: Detailed RoPE vs xPos comparison
    Shows they're nearly identical across all conditions
    """
    print("\nGenerating Figure 4: RoPE vs xPos Comparison...")
    
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)
    
    # Subplot 1: Scatter plot (RoPE vs xPos MASE)
    ax1 = fig.add_subplot(gs[0])
    
    rope_mase = df[df['pe'] == 'rope'].groupby(['dataset', 'L_train', 'H'])['MASE'].mean()
    xpos_mase = df[df['pe'] == 'xpos'].groupby(['dataset', 'L_train', 'H'])['MASE'].mean()
    
    common_idx = rope_mase.index.intersection(xpos_mase.index)
    rope_vals = rope_mase[common_idx].values
    xpos_vals = xpos_mase[common_idx].values
    
    datasets_scatter = [idx[0] for idx in common_idx]
    colors_scatter = [PALETTE_DATASET[d] for d in datasets_scatter]
    
    ax1.scatter(rope_vals, xpos_vals, s=100, alpha=0.7, c=colors_scatter, 
               edgecolors='black', linewidth=1.5)
    
    # Perfect agreement line
    lims = [min(rope_vals.min(), xpos_vals.min()) * 0.95,
            max(rope_vals.max(), xpos_vals.max()) * 1.05]
    ax1.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='Perfect Agreement')
    
    # Correlation
    corr = np.corrcoef(rope_vals, xpos_vals)[0, 1]
    ax1.text(0.05, 0.95, f'r = {corr:.4f}', transform=ax1.transAxes,
            fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.5))
    
    ax1.set_xlabel('RoPE MASE', fontweight='bold')
    ax1.set_ylabel('xPos MASE', fontweight='bold')
    ax1.set_title('RoPE vs xPos Agreement', fontweight='bold')
    ax1.legend()
    
    # Subplot 2: Difference distribution
    ax2 = fig.add_subplot(gs[1])
    
    differences = rope_vals - xpos_vals
    ax2.hist(differences, bins=15, color='steelblue', alpha=0.7, 
            edgecolor='black', linewidth=1.2)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {differences.mean():.4f}')
    ax2.axvline(x=differences.mean(), color='green', linestyle='--', linewidth=2)
    
    ax2.set_xlabel('MASE Difference (RoPE - xPos)', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Performance Difference Distribution', fontweight='bold')
    ax2.legend()
    
    # Subplot 3: By dataset
    ax3 = fig.add_subplot(gs[2])
    
    datasets = sorted(df['dataset'].unique())
    x_pos = np.arange(len(datasets))
    width = 0.35
    
    rope_means = [df[(df['dataset']==d) & (df['pe']=='rope')]['MASE'].mean() 
                 for d in datasets]
    xpos_means = [df[(df['dataset']==d) & (df['pe']=='xpos')]['MASE'].mean()
                 for d in datasets]
    
    ax3.bar(x_pos - width/2, rope_means, width, label='RoPE',
           color=PALETTE_MAIN['rope'], alpha=0.8, edgecolor='black', linewidth=1.2)
    ax3.bar(x_pos + width/2, xpos_means, width, label='xPos',
           color=PALETTE_MAIN['xpos'], alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax3.set_ylabel('MASE', fontweight='bold')
    ax3.set_xlabel('Dataset', fontweight='bold')
    ax3.set_title('RoPE vs xPos by Dataset', fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([d.capitalize() for d in datasets])
    ax3.legend()
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle('RoPE and xPos are Nearly Identical',
                fontweight='bold', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig('reports/figures/fig4_rope_vs_xpos.png')
    plt.savefig('reports/figures/fig4_rope_vs_xpos.pdf')
    print("✓ Saved: fig4_rope_vs_xpos.png")
    plt.close()

def figure5_alibi_failure(df):
    """
    Figure 5: ALiBi failure mode analysis
    Shows why linear decay fails on periodic data
    """
    print("\nGenerating Figure 5: ALiBi Failure Analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: ALiBi vs No-PE comparison
    ax1 = axes[0, 0]
    datasets = sorted(df['dataset'].unique())
    x_pos = np.arange(len(datasets))
    width = 0.35
    
    alibi_mase = [df[(df['dataset']==d) & (df['pe']=='alibi')]['MASE'].mean() 
                 for d in datasets]
    none_mase = [df[(df['dataset']==d) & (df['pe']=='none')]['MASE'].mean()
                for d in datasets]
    
    bars1 = ax1.bar(x_pos - width/2, alibi_mase, width, label='ALiBi',
                   color=PALETTE_MAIN['alibi'], alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x_pos + width/2, none_mase, width, label='No-PE',
                   color=PALETTE_MAIN['none'], alpha=0.8, edgecolor='black')
    
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.6,
               label='Seasonal Naive')
    ax1.set_ylabel('MASE', fontweight='bold')
    ax1.set_xlabel('Dataset', fontweight='bold')
    ax1.set_title('ALiBi vs No Positional Encoding', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([d.capitalize() for d in datasets])
    ax1.legend()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Subplot 2: Performance degradation
    ax2 = axes[0, 1]
    degradation = [(a - n) / n * 100 for a, n in zip(alibi_mase, none_mase)]
    colors_deg = ['red' if d > 0 else 'green' for d in degradation]
    
    bars = ax2.bar(datasets, degradation, color=colors_deg, alpha=0.7,
                  edgecolor='black', linewidth=1.2)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_ylabel('Degradation (%)', fontweight='bold')
    ax2.set_xlabel('Dataset', fontweight='bold')
    ax2.set_title('ALiBi Degradation vs No-PE', fontweight='bold')
    ax2.set_xticklabels([d.capitalize() for d in datasets])
    
    for bar, val in zip(bars, degradation):
        label_y = val + (1 if val > 0 else -2)
        ax2.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{val:+.1f}%', ha='center', va='bottom' if val > 0 else 'top',
                fontsize=10, fontweight='bold')
    
    # Subplot 3: LES comparison (ALiBi vs best)
    ax3 = axes[1, 0]
    les_data = df[df['L_eval'] > df['L_train']]
    
    for pe in ['rope', 'alibi']:
        pe_les = les_data[les_data['pe'] == pe].groupby('dataset')['LES'].mean()
        ax3.plot(pe_les.index, pe_les.values, marker='o', linewidth=2.5,
                markersize=10, label=pe.upper(), color=PALETTE_MAIN[pe])
    
    ax3.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.6,
               label='Perfect')
    ax3.set_ylabel('LES', fontweight='bold')
    ax3.set_xlabel('Dataset', fontweight='bold')
    ax3.set_title('Length Extrapolation: ALiBi vs RoPE', fontweight='bold')
    ax3.set_xticklabels([d.capitalize() for d in ax3.get_xticklabels()])
    ax3.legend()
    
    # Subplot 4: Explanation diagram
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    explanation = """
    Why ALiBi Fails on Time-Series:
    
    ALiBi Assumption:
    • Linear distance penalty: attention(i,j) ∝ -m|i-j|
    • "Older timesteps are less relevant"
    • Monotonic decay with distance
    
    Time-Series Reality:
    • Periodic patterns (24h, 7d cycles)
    • t-24 is HIGHLY relevant (yesterday)
    • t-168 is HIGHLY relevant (last week)
    • Non-monotonic importance
    
    Conclusion:
    • Linear decay contradicts periodicity
    • RoPE's rotary encoding better captures
      temporal structure
    • Domain-specific biases needed
    """
    
    ax4.text(0.1, 0.9, explanation, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('ALiBi Failure Mode: Linear Decay vs Periodic Structure',
                fontweight='bold', fontsize=15, y=0.98)
    plt.tight_layout()
    plt.savefig('reports/figures/fig5_alibi_failure.png')
    plt.savefig('reports/figures/fig5_alibi_failure.pdf')
    print("✓ Saved: fig5_alibi_failure.png")
    plt.close()

def figure6_poster_summary(df):
    """
    Figure 6: Poster-ready summary figure
    One comprehensive figure with all key results
    """
    print("\nGenerating Figure 6: Poster Summary...")
    
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Main result: MASE by PE (averaged across datasets)
    ax_main = fig.add_subplot(gs[0, :2])
    
    overall_mase = df.groupby('pe').agg({
        'MASE': ['mean', 'std']
    })['MASE']
    overall_mase = overall_mase.sort_values('mean')
    
    pes = overall_mase.index
    means = overall_mase['mean'].values
    stds = overall_mase['std'].values
    colors = [PALETTE_MAIN[pe] for pe in pes]
    
    bars = ax_main.bar(range(len(pes)), means, yerr=stds, capsize=7,
                      color=colors, alpha=0.85, edgecolor='black', linewidth=2,
                      error_kw={'linewidth': 2, 'ecolor': 'black'})
    
    ax_main.axhline(y=1.0, color='red', linestyle='--', linewidth=3,
                   alpha=0.7, label='Seasonal Naive Baseline')
    ax_main.set_xticks(range(len(pes)))
    ax_main.set_xticklabels([pe.upper() for pe in pes], fontsize=14)
    ax_main.set_ylabel('MASE (Mean ± Std)', fontweight='bold', fontsize=14)
    ax_main.set_title('Overall Performance Across All Datasets',
                     fontweight='bold', fontsize=16)
    ax_main.legend(fontsize=12)
    ax_main.set_ylim([0, max(means) * 1.2])
    
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax_main.text(i, mean + std + 0.05, f'{mean:.3f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # LES summary
    ax_les = fig.add_subplot(gs[0, 2])
    les_data = df[df['L_eval'] > df['L_train']]
    les_means = les_data.groupby('pe')['LES'].mean().sort_values()
    
    colors_les = [PALETTE_MAIN[pe] for pe in les_means.index]
    ax_les.barh(range(len(les_means)), les_means.values, color=colors_les,
               alpha=0.85, edgecolor='black', linewidth=1.5)
    ax_les.axvline(x=1.0, color='green', linestyle='--', linewidth=2,
                  label='Perfect')
    ax_les.set_yticks(range(len(les_means)))
    ax_les.set_yticklabels([pe.upper() for pe in les_means.index], fontsize=12)
    ax_les.set_xlabel('LES', fontweight='bold', fontsize=12)
    ax_les.set_title('Extrapolation Performance', fontweight='bold', fontsize=14)
    ax_les.legend(fontsize=10)
    
    # Dataset breakdown
    ax_datasets = fig.add_subplot(gs[1, :])
    
    datasets = sorted(df['dataset'].unique())
    x = np.arange(len(datasets))
    width = 0.15
    
    pes_ordered = ['rope', 'xpos', 'absolute', 'alibi', 'none']
    
    for i, pe in enumerate(pes_ordered):
        pe_means = [df[(df['dataset']==d) & (df['pe']==pe)]['MASE'].mean()
                   for d in datasets]
        offset = (i - 2) * width
        ax_datasets.bar(x + offset, pe_means, width, label=pe.upper(),
                       color=PALETTE_MAIN[pe], alpha=0.85, edgecolor='black',
                       linewidth=1)
    
    ax_datasets.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
                       alpha=0.7)
    ax_datasets.set_ylabel('MASE', fontweight='bold', fontsize=14)
    ax_datasets.set_xlabel('Dataset', fontweight='bold', fontsize=14)
    ax_datasets.set_title('Performance Breakdown by Dataset',
                         fontweight='bold', fontsize=16)
    ax_datasets.set_xticks(x)
    ax_datasets.set_xticklabels([d.capitalize() for d in datasets], fontsize=13)
    ax_datasets.legend(ncol=5, fontsize=11, loc='upper right')
    
    plt.suptitle('Positional Encoding for Time-Series Forecasting: Key Results',
                fontweight='bold', fontsize=18, y=0.98)
    plt.savefig('reports/figures/fig6_poster_summary.png')
    plt.savefig('reports/figures/fig6_poster_summary.pdf')
    print("✓ Saved: fig6_poster_summary.png")
    plt.close()

def main():
    """Generate all publication figures."""
    print("\n" + "="*80)
    print("PUBLICATION-QUALITY FIGURE GENERATION")
    print("="*80)
    
    # Load data
    df, stats_df = load_data()
    
    # Create output directory
    import os
    os.makedirs('reports/figures', exist_ok=True)
    
    # Generate all figures
    figure1_mase_comparison(df, stats_df)
    figure2_les_curves(df)
    figure3_multidataset_heatmap(df)
    figure4_rope_vs_xpos(df)
    figure5_alibi_failure(df)
    figure6_poster_summary(df)
    
    print("\n" + "="*80)
    print("ALL FIGURES GENERATED!")
    print("="*80)
    print("\nGenerated 6 publication-quality figures:")
    print("  1. fig1_mase_comparison.png - Main performance comparison")
    print("  2. fig2_les_curves.png - Length extrapolation curves")
    print("  3. fig3_multidataset_heatmap.png - Comprehensive heatmap")
    print("  4. fig4_rope_vs_xpos.png - RoPE vs xPos detailed analysis")
    print("  5. fig5_alibi_failure.png - ALiBi failure mode explanation")
    print("  6. fig6_poster_summary.png - Poster-ready comprehensive summary")
    print("\nAll figures saved in PNG and PDF formats")
    print("Ready for poster presentation and paper submission!")
    print("="*80)

if __name__ == '__main__':
    main()
