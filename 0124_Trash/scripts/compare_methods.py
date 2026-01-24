#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison script for three methods: CNN, Vmamba, SwinT
Extract metrics from training results and create visualization comparisons
"""

import os
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set font for scientific papers (Times New Roman or DejaVu Sans)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def extract_metrics_from_json(json_file):
    """Extract metrics from JSON file"""
    if not os.path.exists(json_file):
        return None
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metrics = {
            'best_auc': data.get('best_auc', 0),
            'best_epoch': data.get('best_epoch', 0),
            'final_val_auc': data.get('val', {}).get('auc', 0),
            'final_val_acc': data.get('val', {}).get('accuracy', 0),
            'final_val_f1': data.get('val', {}).get('f1', 0),
            'final_train_acc': data.get('train', {}).get('accuracy', 0),
        }
        
        # If training_summary.json exists, use it preferentially
        summary_file = json_file.replace('metrics.json', 'training_summary.json')
        if os.path.exists(summary_file):
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
                metrics.update({
                    'best_auc': summary.get('best_val_auc', metrics['best_auc']),
                    'best_epoch': summary.get('best_epoch', metrics['best_epoch']),
                    'final_val_auc': summary.get('final_val_auc', metrics['final_val_auc']),
                    'final_val_acc': summary.get('final_val_acc', metrics['final_val_acc']),
                    'best_val_f1': summary.get('best_val_f1', metrics['final_val_f1']),
                    'final_train_acc': summary.get('final_train_acc', metrics['final_train_acc']),
                })
        
        return metrics
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return None

def extract_metrics_from_log(log_file):
    """Extract best metrics from log file"""
    if not os.path.exists(log_file):
        return None
    
    metrics = {
        'best_auc': 0,
        'best_acc': 0,
        'best_f1': 0,
        'best_epoch': 0,
    }
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # Extract best AUC
            auc_patterns = [
                r'最佳验证AUC[:\s]+(\d+\.?\d*)',
                r'Best.*AUC[:\s]+(\d+\.?\d*)',
                r'best.*auc[:\s]+(\d+\.?\d*)',
                r'Best Validation AUC[:\s]+(\d+\.?\d*)',
            ]
            for pattern in auc_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    try:
                        aucs = [float(x) for x in matches]
                        if aucs:
                            metrics['best_auc'] = max(aucs)
                            break
                    except:
                        pass
            
            # Extract best epoch
            epoch_patterns = [
                r'最佳验证AUC[:\s]+\d+\.?\d*\s+\(Epoch\s+(\d+)\)',
                r'best.*auc.*epoch\s+(\d+)',
                r'Best.*AUC.*\(Epoch\s+(\d+)\)',
                r'best.*epoch[:\s]+(\d+)',
            ]
            for pattern in epoch_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    try:
                        epochs = [int(x) for x in matches]
                        if epochs:
                            metrics['best_epoch'] = max(epochs)
                            break
                    except:
                        pass
    except Exception as e:
        print(f"Error reading log {log_file}: {e}")
    
    return metrics

def collect_all_results():
    """Collect results from all methods"""
    base_dir = Path('/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713')
    
    results = {}
    
    # CNN results
    cnn_dir = base_dir / 'cnn_result'
    if cnn_dir.exists():
        metrics_file = cnn_dir / 'metrics.json'
        log_file = cnn_dir / 'train.log'
        summary_file = cnn_dir / 'training_summary.json'
        
        metrics = extract_metrics_from_json(str(metrics_file))
        if metrics:
            results['CNN'] = metrics
        else:
            log_metrics = extract_metrics_from_log(str(log_file))
            if log_metrics:
                results['CNN'] = log_metrics
    
    # Vmamba results
    vmamba_dir = base_dir / 'vmamba_result'
    if vmamba_dir.exists():
        metrics_file = vmamba_dir / 'metrics.json'
        log_file = vmamba_dir / 'train.log'
        
        metrics = extract_metrics_from_json(str(metrics_file))
        if metrics:
            results['Vmamba'] = metrics
        else:
            log_metrics = extract_metrics_from_log(str(log_file))
            if log_metrics:
                results['Vmamba'] = log_metrics
    
    # SwinT results
    swint_dir = base_dir / 'models' / 'SwinT' / '_results' / 'multimodal'
    if swint_dir.exists():
        metrics_file = swint_dir / 'metrics.json'
        log_file = swint_dir / 'train.log'
        summary_file = swint_dir / 'training_summary.json'
        
        metrics = extract_metrics_from_json(str(metrics_file))
        if metrics:
            results['SwinT'] = metrics
        else:
            log_metrics = extract_metrics_from_log(str(log_file))
            if log_metrics:
                results['SwinT'] = log_metrics
    
    return results

def create_comparison_plots(results, output_dir):
    """Create comparison plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    methods = list(results.keys())
    best_aucs = [results[m].get('best_auc', 0) for m in methods]
    final_val_accs = [results[m].get('final_val_acc', 0) for m in methods]
    final_val_f1s = [results[m].get('final_val_f1', 0) for m in methods]
    final_val_aucs = [results[m].get('final_val_auc', 0) for m in methods]
    
    # 1. Best AUC comparison bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, best_aucs, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.title('Best Validation AUC Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('AUC', fontsize=12)
    plt.xlabel('Method', fontsize=12)
    plt.ylim([0, 1.0])
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, best_aucs)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_auc_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/best_auc_comparison.png")
    
    # 2. Multi-metric radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    categories = ['Best AUC', 'Val Accuracy', 'Val F1', 'Final AUC']
    num_vars = len(categories)
    
    # Calculate angles
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    # Plot for each method
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for idx, method in enumerate(methods):
        values = [
            best_aucs[idx],
            final_val_accs[idx] / 100.0,  # Convert to 0-1 range
            final_val_f1s[idx],
            final_val_aucs[idx]
        ]
        values += values[:1]  # Close the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True)
    
    plt.title('Multi-Metric Comprehensive Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/radar_comparison.png")
    
    # 3. Multi-metric comparison bar chart group
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Metrics Comparison of Three Methods', fontsize=16, fontweight='bold', y=0.995)
    
    # Best AUC
    axes[0, 0].bar(methods, best_aucs, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='black')
    axes[0, 0].set_title('Best Validation AUC', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('AUC', fontsize=10)
    axes[0, 0].set_ylim([0, 1.0])
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(best_aucs):
        axes[0, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Validation Accuracy
    axes[0, 1].bar(methods, final_val_accs, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='black')
    axes[0, 1].set_title('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=10)
    axes[0, 1].set_ylim([0, 100])
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(final_val_accs):
        axes[0, 1].text(i, v + 2, f'{v:.2f}%', ha='center', va='bottom', fontsize=9)
    
    # Validation F1
    axes[1, 0].bar(methods, final_val_f1s, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='black')
    axes[1, 0].set_title('Validation F1-Score', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('F1-Score', fontsize=10)
    axes[1, 0].set_ylim([0, 1.0])
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(final_val_f1s):
        axes[1, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Final AUC
    axes[1, 1].bar(methods, final_val_aucs, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='black')
    axes[1, 1].set_title('Final Validation AUC', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('AUC', fontsize=10)
    axes[1, 1].set_ylim([0, 1.0])
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(final_val_aucs):
        axes[1, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multi_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/multi_metrics_comparison.png")

def create_comparison_table(results, output_dir):
    """Create comparison table"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    data = []
    for method, metrics in results.items():
        data.append({
            'Method': method,
            'Best AUC': f"{metrics.get('best_auc', 0):.4f}",
            'Best Epoch': metrics.get('best_epoch', 0),
            'Val Accuracy (%)': f"{metrics.get('final_val_acc', 0):.2f}",
            'Val F1-Score': f"{metrics.get('final_val_f1', 0):.4f}",
            'Final AUC': f"{metrics.get('final_val_auc', 0):.4f}",
            'Train Accuracy (%)': f"{metrics.get('final_train_acc', 0):.2f}",
        })
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    csv_file = os.path.join(output_dir, 'comparison_table.csv')
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"✓ Saved: {csv_file}")
    
    # Save as Markdown table
    md_file = os.path.join(output_dir, 'comparison_table.md')
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# Performance Comparison Table of Three Methods\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Description\n\n")
        f.write("- **Best AUC**: Highest AUC value on validation set during training\n")
        f.write("- **Best Epoch**: Epoch at which best AUC was achieved\n")
        f.write("- **Val Accuracy**: Classification accuracy on validation set\n")
        f.write("- **Val F1-Score**: F1 score on validation set\n")
        f.write("- **Final AUC**: Validation AUC at the last epoch\n")
        f.write("- **Train Accuracy**: Accuracy on training set\n")
    print(f"✓ Saved: {md_file}")
    
    # Print table
    print("\n" + "="*80)
    print("Performance Comparison Table of Three Methods")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    return df

def main():
    """Main function"""
    print("Collecting training results from three methods...")
    print("="*80)
    
    results = collect_all_results()
    
    if not results:
        print("No training results found. Please run training scripts first.")
        return
    
    print(f"\nFound results from {len(results)} methods:")
    for method in results.keys():
        print(f"  - {method}")
    
    # Create output directory
    output_dir = Path('/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713') / 'comparison_results'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating comparison plots...")
    create_comparison_plots(results, str(output_dir))
    
    print("\nGenerating comparison table...")
    create_comparison_table(results, str(output_dir))
    
    print(f"\nAll comparison results saved to: {output_dir}")
    print("\nComparison completed!")

if __name__ == '__main__':
    main()

