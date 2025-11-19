"""
Multi-Seed Statistical Analysis for Time-Series Positional Encoding Study
==========================================================================

This script performs comprehensive statistical analysis on multi-seed experimental
results, including:
- Descriptive statistics with confidence intervals
- Statistical significance testing (t-tests, effect sizes)
- ANOVA for multi-group comparisons
- Publication-ready summary tables

Author: DS1011 Final Project
Date: November 2025
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, ttest_rel, f_oneway
import warnings
warnings.filterwarnings('ignore')

def load_and_validate_data(filepath='reports/tables/extrapolation.csv'):
    """Load and validate experimental results."""
    print("="*80)
    print("LOADING AND VALIDATING DATA")
    print("="*80)
    
    df = pd.read_csv(filepath)
    
    print(f"\nTotal evaluations: {len(df)}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"PEs tested: {sorted(df['pe'].unique())}")
    print(f"Horizons: {sorted(df['H'].unique())}")
    print(f"Training lengths: {sorted(df['L_train'].unique())}")
    
    # Check for seeds
    if 'seed' in df.columns:
        print(f"Seeds: {sorted(df['seed'].unique())}")
        seeds_per_config = df.groupby(['dataset', 'pe', 'L_train', 'H'])['seed'].nunique()
        print(f"Seeds per config: {seeds_per_config.min()}-{seeds_per_config.max()}")
    else:
        print("⚠ WARNING: No seed column found. Single-seed analysis only.")
        df['seed'] = 42  # Add dummy seed
    
    return df

def compute_statistics(df):
    """Compute comprehensive statistics with confidence intervals."""
    print("\n" + "="*80)
    print("COMPUTING STATISTICS")
    print("="*80)
    
    # Group by all non-seed variables
    group_cols = ['dataset', 'pe', 'L_train', 'H', 'L_eval']
    
    stats_list = []
    
    for group_key, group_data in df.groupby(group_cols):
        n = len(group_data)
        
        # Compute stats for each metric
        for metric in ['MAE', 'RMSE', 'sMAPE', 'MASE', 'LES']:
            values = group_data[metric].values
            
            mean = np.mean(values)
            std = np.std(values, ddof=1) if n > 1 else 0
            sem = std / np.sqrt(n) if n > 1 else 0
            ci_95 = 1.96 * sem
            
            stats_dict = dict(zip(group_cols, group_key))
            stats_dict.update({
                'metric': metric,
                'n': n,
                'mean': mean,
                'std': std,
                'sem': sem,
                'ci_95': ci_95,
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            })
            stats_list.append(stats_dict)
    
    stats_df = pd.DataFrame(stats_list)
    
    print(f"\n✓ Computed statistics for {len(stats_df)} configurations")
    return stats_df

def test_significance(df):
    """Perform statistical significance tests."""
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)
    
    results = []
    
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset]
        
        # Test RoPE vs each other PE
        rope_data = dataset_data[dataset_data['pe'] == 'rope']
        
        for other_pe in ['absolute', 'xpos', 'alibi', 'none']:
            other_data = dataset_data[dataset_data['pe'] == other_pe]
            
            if len(other_data) == 0:
                continue
            
            # Try to match on L_eval and seed for paired test
            merged = pd.merge(
                rope_data[['L_train', 'H', 'L_eval', 'seed', 'MASE']],
                other_data[['L_train', 'H', 'L_eval', 'seed', 'MASE']],
                on=['L_train', 'H', 'L_eval', 'seed'],
                suffixes=('_rope', f'_{other_pe}'),
                how='inner'
            )
            
            if len(merged) >= 3:
                # Paired t-test
                t_stat, p_value = ttest_rel(merged['MASE_rope'], merged[f'MASE_{other_pe}'])
                test_type = 'paired'
            elif len(rope_data) >= 3 and len(other_data) >= 3:
                # Independent t-test
                t_stat, p_value = ttest_ind(rope_data['MASE'], other_data['MASE'])
                test_type = 'independent'
            else:
                continue
            
            # Effect size (Cohen's d)
            rope_mean = rope_data['MASE'].mean()
            other_mean = other_data['MASE'].mean()
            pooled_std = np.sqrt((rope_data['MASE'].std()**2 + other_data['MASE'].std()**2) / 2)
            cohens_d = (rope_mean - other_mean) / pooled_std if pooled_std > 0 else 0
            
            # Improvement percentage
            improvement = (other_mean - rope_mean) / other_mean * 100
            
            results.append({
                'dataset': dataset,
                'comparison': f'RoPE vs {other_pe.upper()}',
                'test_type': test_type,
                'rope_mean': rope_mean,
                'other_mean': other_mean,
                'difference': rope_mean - other_mean,
                'improvement_pct': improvement,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'n_rope': len(rope_data),
                'n_other': len(other_data),
                'significant': '✓' if p_value < 0.05 else '✗',
                'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
            })
    
    sig_df = pd.DataFrame(results)
    
    if len(sig_df) > 0:
        print("\nSignificance Test Summary:")
        print(sig_df[['dataset', 'comparison', 'improvement_pct', 'p_value', 'cohens_d', 'significant']].to_string(index=False))
    
    return sig_df

def anova_test(df):
    """ANOVA test to compare all PEs simultaneously."""
    print("\n" + "="*80)
    print("ANOVA: COMPARING ALL PEs")
    print("="*80)
    
    results = []
    
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset]
        
        # Group MASE values by PE
        groups = [group_data['MASE'].values for pe, group_data in dataset_data.groupby('pe')]
        
        if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
            f_stat, p_value = f_oneway(*groups)
            
            results.append({
                'dataset': dataset,
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': '✓' if p_value < 0.05 else '✗',
                'n_groups': len(groups)
            })
    
    anova_df = pd.DataFrame(results)
    
    if len(anova_df) > 0:
        print("\nANOVA Results:")
        print(anova_df.to_string(index=False))
        print("\nInterpretation: Significant ANOVA (p<0.05) indicates that at least one PE")
        print("                differs significantly from others.")
    
    return anova_df

def create_summary_tables(df, stats_df):
    """Create publication-ready summary tables."""
    print("\n" + "="*80)
    print("CREATING SUMMARY TABLES")
    print("="*80)
    
    # Table 1: Overall performance by dataset and PE
    table1 = stats_df[stats_df['metric'] == 'MASE'].groupby(['dataset', 'pe']).agg({
        'mean': 'mean',
        'ci_95': 'mean',
        'n': 'first'
    }).reset_index()
    
    table1['MASE_str'] = table1.apply(
        lambda x: f"{x['mean']:.3f} ± {x['ci_95']:.3f}", axis=1
    )
    
    table1_pivot = table1.pivot(index='pe', columns='dataset', values='MASE_str')
    
    print("\nTable 1: MASE by PE and Dataset (Mean ± 95% CI)")
    print(table1_pivot)
    table1_pivot.to_csv('reports/tables/table1_mase_summary.csv')
    print("✓ Saved: reports/tables/table1_mase_summary.csv")
    
    # Table 2: LES performance
    table2 = stats_df[stats_df['metric'] == 'LES'].groupby(['dataset', 'pe']).agg({
        'mean': 'mean',
        'ci_95': 'mean'
    }).reset_index()
    
    table2['LES_str'] = table2.apply(
        lambda x: f"{x['mean']:.4f} ± {x['ci_95']:.4f}", axis=1
    )
    
    table2_pivot = table2.pivot(index='pe', columns='dataset', values='LES_str')
    
    print("\nTable 2: LES by PE and Dataset (Mean ± 95% CI)")
    print(table2_pivot)
    table2_pivot.to_csv('reports/tables/table2_les_summary.csv')
    print("✓ Saved: reports/tables/table2_les_summary.csv")
    
    # Table 3: Best configuration per dataset
    best_configs = []
    for dataset in df['dataset'].unique():
        dataset_stats = stats_df[(stats_df['dataset'] == dataset) & (stats_df['metric'] == 'MASE')]
        best_pe = dataset_stats.groupby('pe')['mean'].mean().idxmin()
        best_row = dataset_stats[dataset_stats['pe'] == best_pe].iloc[0]
        
        best_configs.append({
            'Dataset': dataset.capitalize(),
            'Best PE': best_pe.upper(),
            'MASE': f"{best_row['mean']:.3f} ± {best_row['ci_95']:.3f}",
            'vs Naive': f"{(1.0 - best_row['mean']) * 100:+.1f}%"
        })
    
    table3 = pd.DataFrame(best_configs)
    print("\nTable 3: Best Configuration per Dataset")
    print(table3.to_string(index=False))
    table3.to_csv('reports/tables/table3_best_configs.csv', index=False)
    print("✓ Saved: reports/tables/table3_best_configs.csv")
    
    return table1_pivot, table2_pivot, table3

def generate_latex_table(stats_df):
    """Generate LaTeX table for paper."""
    print("\n" + "="*80)
    print("GENERATING LaTeX TABLE")
    print("="*80)
    
    # Create main results table
    latex = "\\begin{table}[t]\n"
    latex += "\\centering\n"
    latex += "\\caption{Forecasting performance across datasets and positional encodings. "
    latex += "MASE < 1.0 indicates better than seasonal naive baseline. "
    latex += "Results shown as mean $\\pm$ 95\\% confidence interval.}\n"
    latex += "\\label{tab:main_results}\n"
    latex += "\\begin{tabular}{llccc}\n"
    latex += "\\toprule\n"
    latex += "Dataset & PE & MASE & LES & vs Naive \\\\\n"
    latex += "\\midrule\n"
    
    for dataset in sorted(stats_df['dataset'].unique()):
        dataset_data = stats_df[stats_df['dataset'] == dataset]
        
        # Get best PE
        mase_data = dataset_data[dataset_data['metric'] == 'MASE']
        best_pe = mase_data.groupby('pe')['mean'].mean().idxmin()
        
        for pe in sorted(dataset_data['pe'].unique()):
            pe_mase = mase_data[mase_data['pe'] == pe].iloc[0]
            
            les_data = dataset_data[(dataset_data['metric'] == 'LES') & (dataset_data['pe'] == pe)]
            if len(les_data) > 0:
                pe_les = les_data.iloc[0]
            else:
                continue
            
            vs_naive = (1.0 - pe_mase['mean']) * 100
            
            # Bold if best
            pe_str = f"\\textbf{{{pe.upper()}}}" if pe == best_pe else pe.upper()
            
            latex += f"{dataset.capitalize()} & {pe_str} & "
            latex += f"{pe_mase['mean']:.2f}$\\pm${pe_mase['ci_95']:.2f} & "
            latex += f"{pe_les['mean']:.3f}$\\pm${pe_les['ci_95']:.3f} & "
            latex += f"{vs_naive:+.1f}\\% \\\\\n"
        
        latex += "\\midrule\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    # Save
    with open('reports/tables/latex_main_table.tex', 'w') as f:
        f.write(latex)
    
    print("✓ Saved: reports/tables/latex_main_table.tex")

def main():
    """Main analysis pipeline."""
    print("\n" + "="*80)
    print("MULTI-SEED STATISTICAL ANALYSIS")
    print("Time-Series Positional Encoding Study")
    print("="*80)
    
    # Load data
    df = load_and_validate_data()
    
    # Compute statistics
    stats_df = compute_statistics(df)
    stats_df.to_csv('reports/tables/comprehensive_statistics.csv', index=False)
    print("\n✓ Saved: reports/tables/comprehensive_statistics.csv")
    
    # Significance tests
    sig_df = test_significance(df)
    if len(sig_df) > 0:
        sig_df.to_csv('reports/tables/significance_tests.csv', index=False)
        print("✓ Saved: reports/tables/significance_tests.csv")
    
    # ANOVA
    anova_df = anova_test(df)
    if len(anova_df) > 0:
        anova_df.to_csv('reports/tables/anova_results.csv', index=False)
        print("✓ Saved: reports/tables/anova_results.csv")
    
    # Summary tables
    create_summary_tables(df, stats_df)
    
    # LaTeX table
    generate_latex_table(stats_df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nNext: Run create_figures.py to generate visualizations")

if __name__ == '__main__':
    main()
