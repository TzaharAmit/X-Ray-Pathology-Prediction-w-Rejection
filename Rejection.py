#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from sklearn.metrics import precision_recall_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from joblib import Parallel, delayed

# Dictionary for mapping category indices to diagnosis names
CATEGORY_MAPPING = {
    0: "Cardiomegaly",
    1: "Effusion",
    2: "Edema",
    3: "Consolidation"
}

# Colors for consistent plotting
CATEGORY_COLORS = {
    "0": "#1f77b4",  # blue
    "1": "#ff7f0e",  # orange
    "2": "#2ca02c",  # green
    "3": "#d62728",  # red
}

def setup_results_directory(base_path):
    """Create results directory if it doesn't exist"""
    results_dir = os.path.join(base_path, 'results', 'model_performance')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def trimmed_variance(data, proportiontocut):
    """Calculate variance of trimmed data"""
    n = len(data)
    k = int(proportiontocut * n)
    sorted_data = np.sort(data)
    trimmed_data = sorted_data[k:n-k]
    return np.var(trimmed_data, ddof=1)

# Add to scipy.stats namespace
stats.trimmed_variance = trimmed_variance

def compute_distribution_optimized(data, labels, dataset_name):
    """Vectorized distribution computation with improved statistics - only for correctly identified samples"""
    bins = np.linspace(0, 1, 11)
    summaries = []

    for col in data.columns:
        # Get the samples where the model correctly identified the category
        # For positive cases: sample is positive (label=1) and model output is high
        # For negative cases: sample is negative (label=0) and model output is low
        correctly_identified_mask = (
            # True negatives
            ((labels[col] == 0) & (data[col] < 0.5)) | 
            # True positives
            ((labels[col] == 1) & (data[col] >= 0.5))
        )
        
        # Filter values to only include correctly identified samples
        values = data[col][correctly_identified_mask]
        
        # If no correctly identified samples, use all data (fallback)
        if len(values) == 0:
            print(f"Warning: No correctly identified samples for {col} in {dataset_name}. Using all samples.")
            values = data[col]
        
        # Check for skewness to decide on appropriate central tendency
        skewness = stats.skew(values)
        kurtosis = stats.kurtosis(values)
        
        # Compute histogram
        hist, bin_edges = np.histogram(values, bins=bins)
        max_bin_idx = np.argmax(hist)
        most_common_range = (bins[max_bin_idx], bins[max_bin_idx + 1])
        
        # Choose appropriate central tendency based on distribution
        if abs(skewness) > 1.0:  # If distribution is notably skewed
            central_value = stats.trim_mean(values, 0.1)  # 10% trimmed mean
            std_dev = np.sqrt(stats.trimmed_variance(values, 0.1))
        else:
            central_value = np.median(values)
            mad = stats.median_abs_deviation(values)
            std_dev = mad * 1.4826  
        
        # Calculate correctly identified percentage using the round function
        correctly_identified_pct = round(100 * len(values) / len(data[col]), 1)
        
        summaries.append({
            "Dataset": dataset_name,
            "Category": col, 
            "Category Name": CATEGORY_MAPPING[int(col)],
            "Median": central_value,
            "Std Dev": std_dev,
            "Skewness": skewness,
            "Kurtosis": kurtosis,
            "Most Common Probability Range": f"[{most_common_range[0]:.2f}, {most_common_range[1]:.2f}]",
            "Count": hist[max_bin_idx],
            "Correctly Identified %": correctly_identified_pct
        })

    df = pd.DataFrame(summaries)
    return df

def optimize_threshold_for_category(col, data, labels, prob_range, median, std_dev):
    """Optimized threshold finding for a single category using distribution-aware metrics"""
    # Adjust the range of thresholds based on the distribution
    # For more skewed distributions, we might want to try a wider range
    skewness = abs(stats.skew(data[col]))
    
    # Adjust threshold range based on skewness
    if skewness > 1.5:  # Highly skewed
        threshold_multipliers = np.linspace(0.1, 4.0, 25)
    elif skewness > 0.5:  # Moderately skewed
        threshold_multipliers = np.linspace(0.15, 3.5, 20)
    else:  # More symmetric
        threshold_multipliers = np.linspace(0.2, 3.0, 20)
    
    possible_thresholds = std_dev * threshold_multipliers

    category_results = []

    for th in possible_thresholds:
        abs_deviations = np.abs(data[col] - median)
        predictions = (abs_deviations > th).astype(int)
        
        # Calculate rejection rate
        rejected_samples = (abs_deviations <= th).sum()
        rejection_rate = rejected_samples / len(data[col])

        f1 = f1_score(labels[col], predictions, zero_division=0)
        precision = precision_score(labels[col], predictions, zero_division=0)
        recall = recall_score(labels[col], predictions, zero_division=0)

        precisions, recalls, _ = precision_recall_curve(labels[col], data[col])
        pr_auc = auc(recalls, precisions) if len(precisions) > 0 and len(recalls) > 0 else 0

        category_results.append({
            'Category': col,
            'Category Name': CATEGORY_MAPPING[int(col)],
            'Threshold': th,
            'F1 Score': f1,
            'Precision': precision,
            'Recall': recall,
            'Precision-Recall AUC': pr_auc,
            'Median': median,
            'Std Dev': std_dev,
            'Rejection Rate': rejection_rate
        })

     # Find best result - using a weighted sum approach
    if skewness > 1.0:
        weights = {'F1 Score': 0.50, 'Precision-Recall AUC': 0.45, 'Recall': 0.05}
    else:
        weights = {'F1 Score': 0.00, 'Precision-Recall AUC': 0.95, 'Recall': 0.05}

    # Add weighted score column to results
    for result in category_results:
        result['Weighted Score'] = (
            (result['F1 Score'] * weights['F1 Score']) + 
            (result['Precision-Recall AUC'] * weights['Precision-Recall AUC']) + 
            (result['Recall'] * weights['Recall'])
            )

    best_result = max(category_results, key=lambda x: x['Weighted Score'])

    return category_results, best_result

def plot_combined_rejection_vs_weighted_score(all_category_results, dataset_name, results_dir):
    """Plot rejection rate versus weighted score for all categories in one plot"""
    plt.figure(figsize=(12, 8))
    
    # Dictionary to store best points for annotation
    best_points = {}
    
    # Plot each category
    for category, results in all_category_results.items():
        df = pd.DataFrame(results)
        category_name = CATEGORY_MAPPING[int(category)]
        color = CATEGORY_COLORS[category]
        
        # Plot the curve
        plt.plot(df['Rejection Rate'], df['Weighted Score'], 'o-', 
                 linewidth=2, color=color, label=category_name)
        
        # Find and store the best point
        best_point = df.loc[df['Weighted Score'].idxmax()]
        best_points[category] = best_point
        
        # Add marker for best point
        plt.scatter(best_point['Rejection Rate'], best_point['Weighted Score'], 
                    color=color, s=100, edgecolor='black')
    
    # Add annotations for best points
    for category, point in best_points.items():
        category_name = CATEGORY_MAPPING[int(category)]
        plt.annotate(f"{category_name}: TH={point['Threshold']:.4f}\nScore={point['Weighted Score']:.3f}, RR={point['Rejection Rate']:.2f}",
                     xy=(point['Rejection Rate'], point['Weighted Score']),
                     xytext=(10, 10),
                     textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", color='black'))
    
    plt.title(f'{dataset_name} Dataset\nRejection Rate vs. Weighted Score for All Categories')
    plt.xlabel('Rejection Rate')
    plt.ylabel('Weighted Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Categories", loc='lower right')
    
    # Save plot
    plot_path = os.path.join(results_dir, f'{dataset_name}_combined_rejection_vs_weighted_score.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created combined plot for {dataset_name}: {plot_path}")
    
    return plot_path


def find_optimal_thresholds_parallel(data, labels, distributions, dataset_name, results_dir):
    """Parallel processing of threshold optimization with dataset name and formatted table"""
    optimal_results = []
    all_category_results = {}

    for _, row in distributions.iterrows():
        category = row['Category']
        category_results, best_result = optimize_threshold_for_category(
            category,
            data,
            labels,
            eval(row['Most Common Probability Range']),
            row['Median'],
            row['Std Dev']  # Pass the std_dev computed based on distribution
        )
        
        optimal_results.append(best_result)
        all_category_results[category] = category_results

    # Create combined plot with all categories
    plot_path = plot_combined_rejection_vs_weighted_score(all_category_results, dataset_name, results_dir)

    optimal_thresholds = pd.DataFrame(optimal_results)

   # Create summary table with dataset name
    summary_table = optimal_thresholds[['Category', 'Category Name', 'Threshold', 'Rejection Rate', 'F1 Score', 'Weighted Score']].copy()
    summary_table.rename(columns={'Threshold': 'Optimal TH'}, inplace=True)
    summary_table.insert(0, 'Dataset', dataset_name)

    # Display formatted table using Pandas styling
    from IPython.display import display
    styled_table = summary_table.style.set_properties(**{
    'border': '1px solid black', 
    'text-align': 'center'
    }).set_table_styles([
        {'selector': 'th', 'props': [('border', '1px solid black'), ('text-align', 'center'), ('background-color', '#f2f2f2')]},
        {'selector': 'td', 'props': [('border', '1px solid black'), ('text-align', 'center')]}
    ]).format({
        'Optimal TH': '{:.4f}',
        'Rejection Rate': '{:.2f}',
        'F1 Score': '{:.3f}',
        'Weighted Score': '{:.3f}'
    })

    print(f"\nOptimal Threshold Summary for {dataset_name}:")
    display(styled_table)
    return optimal_thresholds

def calculate_average_thresholds(cx_optimal_thresholds, nih_optimal_thresholds, pc_optimal_thresholds):
    """Calculate the average threshold for each pathology across all datasets"""
    # Combine all thresholds
    all_thresholds = pd.concat([
        cx_optimal_thresholds[['Category', 'Category Name', 'Threshold', 'Rejection Rate', 'F1 Score', 'Weighted Score']],
        nih_optimal_thresholds[['Category', 'Category Name', 'Threshold', 'Rejection Rate', 'F1 Score', 'Weighted Score']],
        pc_optimal_thresholds[['Category', 'Category Name', 'Threshold', 'Rejection Rate', 'F1 Score', 'Weighted Score']]
    ])
    
    # Group by Category and compute average values
    avg_thresholds = all_thresholds.groupby(['Category', 'Category Name']).agg({
        'Threshold': 'mean',
        'Rejection Rate': 'mean',
        'F1 Score': 'mean',
        'Weighted Score': 'mean'
    }).reset_index()
    
    print("\nAverage Optimal Thresholds Across All Datasets:")
    
    # Display formatted table using Pandas styling
    from IPython.display import display
    styled_table = avg_thresholds.style.set_properties(**{
        'border': '1px solid black', 
        'text-align': 'center'
    }).set_table_styles([
        {'selector': 'th', 'props': [('border', '1px solid black'), ('text-align', 'center'), ('background-color', '#f2f2f2')]},
        {'selector': 'td', 'props': [('border', '1px solid black'), ('text-align', 'center')]}
    ]).format({
        'Threshold': '{:.4f}',
        'Rejection Rate': '{:.2f}',
        'F1 Score': '{:.3f}',
        'Weighted Score': '{:.3f}'
    })
    
    display(styled_table)
    
    return avg_thresholds

def apply_rejection_mechanism_with_avg_thresholds(data, labels, avg_thresholds):
    """Apply rejection mechanism using average thresholds and compute performance metrics"""
    rejection_results = []
    
    # Create arrays to store all non-rejected predictions and labels
    all_predictions = []
    all_labels = []

    for _, row in avg_thresholds.iterrows():
        col = row['Category']
        th = row['Threshold']
        skewness = stats.skew(data[col])  # Compute skewness for the category
        
        # Calculate median value for thresholding
        median_value = np.median(data[col])

        # Calculate absolute deviations
        abs_deviations = np.abs(data[col] - median_value)
        
        # Apply different rejection logic based on skewness
        if skewness > 1.0:
            # Right-skewed distribution: reject only low values
            non_rejected_mask = data[col] >= (median_value - th)
        elif skewness < -1.0:
            # Left-skewed distribution: reject only high values
            non_rejected_mask = data[col] <= (median_value + th)
        else:
            # Symmetric distribution: reject based on absolute deviation
            non_rejected_mask = abs_deviations > th
        
        # Keep track of predictions and true labels for non-rejected samples
        all_predictions.extend(data[col][non_rejected_mask].tolist())
        all_labels.extend(labels[col][non_rejected_mask].tolist())
        
        # Create binary predictions
        predictions = (~non_rejected_mask).astype(int)
        
        # Calculate rejection rate
        rejected_samples = (~non_rejected_mask).sum()
        rejection_rate = rejected_samples / len(data[col])

        # Compute performance metrics
        auc_score = roc_auc_score(labels[col], data[col])
        accuracy = accuracy_score(labels[col], predictions)
        precision = precision_score(labels[col], predictions, zero_division=0)
        recall = recall_score(labels[col], predictions, zero_division=0)
        f1 = f1_score(labels[col], predictions, zero_division=0)

        rejection_results.append({
            'Category': col,
            'Category Name': CATEGORY_MAPPING[int(col)],
            'AUC': auc_score,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Rejection Rate': rejection_rate
        })

    return pd.DataFrame(rejection_results), np.array(all_predictions), np.array(all_labels)
def compute_performance_table(rejection_performance):
    """
    Compute the formatted table displaying AUC performance after applying the rejection mechanism.
    """
    # Define category mapping and order
    CATEGORY_MAPPING = {
        "0": "Cardiomegaly",
        "1": "Effusion",
        "2": "Edema",
        "3": "Consolidation"
    }
    CATEGORY_ORDER = ["Avg Test AUC", "Cardiomegaly", "Effusion", "Edema", "Consolidation"]
    
    # Select relevant columns
    df_results = rejection_performance[['Dataset', 'Category', 'AUC']].copy()
    
    # Replace category numbers with names
    df_results['Category'] = df_results['Category'].astype(str).map(CATEGORY_MAPPING)
    
    # Compute mean and standard deviation for each category
    mean_std = df_results.groupby("Category")["AUC"].agg(["mean", "std"]).reset_index()
    mean_std["AUC"] = mean_std["mean"].round(2).astype(str) + " ± " + mean_std["std"].round(2).astype(str)
    mean_std = mean_std.drop(columns=["mean", "std"])
    mean_std["Dataset"] = "MEAN"
    
    # Compute overall average test AUC per dataset
    avg_test_auc = df_results.groupby("Dataset")["AUC"].mean().reset_index()
    avg_test_auc["Category"] = "Avg Test AUC"
    
    # Compute overall mean for MEAN column
    overall_mean = df_results["AUC"].mean()
    overall_std = df_results["AUC"].std()
    overall_row = pd.DataFrame({
        "Dataset": ["MEAN"],
        "Category": ["Avg Test AUC"],
        "AUC": [f"{overall_mean:.2f} ± {overall_std:.2f}"]
    })
    
    # Append mean row and avg test AUC row to the results DataFrame
    df_results = pd.concat([df_results, mean_std, avg_test_auc, overall_row], ignore_index=True)
    
    # Pivot the table to match the required format and ensure MEAN is the rightmost column
    df_pivot = df_results.pivot(index="Category", columns="Dataset", values="AUC")
    
    # Reorder rows to match the defined category order
    df_pivot = df_pivot.reindex(CATEGORY_ORDER)
    
    # Reorder columns to ensure MEAN is the rightmost
    columns_order = [col for col in df_pivot.columns if col != "MEAN"] + ["MEAN"]
    df_pivot = df_pivot[columns_order]
    
    # Format table with borders
    from IPython.display import display
    styled_table = df_pivot.style.set_properties(
    **{'border': '1px solid black', 'text-align': 'center'}
    ).set_table_styles([
    {'selector': 'th', 'props': [('border', '1px solid black'), ('text-align', 'center'), ('background-color', '#f2f2f2')]},
    {'selector': 'td', 'props': [('border', '1px solid black'), ('text-align', 'center')]}
    ])

    display(styled_table)
    return df_pivot

def plot_overall_roc_curve(all_predictions, all_labels, results_dir, baseline_auc=0.7886):
    """Create a single ROC curve for overall model performance with baseline comparison"""
    plt.figure(figsize=(10, 8))
    
    # Calculate ROC curve and AUC for rejection model
    fpr, tpr, _ = roc_curve(all_labels, all_predictions)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve for model with rejection
    plt.plot(fpr, tpr, color='blue', lw=2, 
             label=f'With Rejection (AUC = {roc_auc:.3f})')
    
    # Create a simulated baseline ROC curve based on the AUC value
    # This is an approximation as we don't have the actual FPR/TPR points
    # Using a simple parametric model of ROC curve based on AUC
    x = np.linspace(0, 1, 100)
    # Formula for ROC curve with given AUC (binormal model approximation)
    a = np.sqrt(2) * stats.norm.ppf(baseline_auc)
    y = stats.norm.cdf(a - stats.norm.ppf(1-x))
    
    plt.plot(x, y, color='red', lw=2, linestyle='--',
             label=f'Baseline Model (AUC = {baseline_auc:.3f})')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.7)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Overall ROC Curve Comparison: Baseline vs. Rejection Model')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add improvement text
    improvement = roc_auc - baseline_auc
    plt.annotate(f'Improvement: {improvement:.3f}', 
                 xy=(0.5, 0.1), 
                 xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Save plot
    roc_plot_path = os.path.join(results_dir, 'overall_roc_curve_comparison.png')
    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created comparative ROC curve: {roc_plot_path}")
    
    return roc_plot_path


def display_rejection_performance(cx_rejection_performance, nih_rejection_performance, pc_rejection_performance):
    """Display the rejection performance results and calculate averages matching the screenshot format"""
    # Combine all results
    combined_results = pd.concat([
        cx_rejection_performance.assign(Dataset='CX'),
        nih_rejection_performance.assign(Dataset='NIH'),
        pc_rejection_performance.assign(Dataset='PC')
    ])
    
    # Create a new table focused only on AUC values which will match the screenshot
    auc_table = combined_results[['Dataset', 'Category', 'Category Name', 'AUC']].copy()
    
    # Calculate average AUC per dataset (across all categories)
    dataset_avg = auc_table.groupby('Dataset')['AUC'].mean().reset_index()
    dataset_avg['Category'] = '0'  # Placeholder
    dataset_avg['Category Name'] = 'Avg Test AUC'
    
    # Calculate average AUC per category (across all datasets)
    category_avg = auc_table.groupby(['Category', 'Category Name'])['AUC'].agg(['mean', 'std']).reset_index()
    category_avg['mean'] = category_avg['mean'].apply(lambda x: round(x, 2))
    category_avg['std'] = category_avg['std'].apply(lambda x: round(x, 2))
    category_avg['AUC'] = category_avg['mean'].astype(str) + " ± " + category_avg['std'].astype(str)
    category_avg['Dataset'] = 'MEAN'
    category_avg = category_avg[['Dataset', 'Category', 'Category Name', 'AUC']]
    
    # Calculate overall average (across all datasets and categories)
    overall_mean = round(auc_table['AUC'].mean(), 2)
    overall_std = round(auc_table['AUC'].std(), 2)
    overall_avg = pd.DataFrame({
        'Dataset': ['MEAN'],
        'Category': ['0'],
        'Category Name': ['Avg Test AUC'],
        'AUC': [f"{overall_mean} ± {overall_std}"]
    })
    
    # Combine all rows
    final_table = pd.concat([auc_table, dataset_avg, category_avg, overall_avg])
    
    # Prepare the pivot table
    pivot_table = final_table.pivot(index='Category Name', columns='Dataset', values='AUC').reset_index()
    
    # Reorder the rows to match the screenshot
    row_order = ['Avg Test AUC', 'Cardiomegaly', 'Effusion', 'Edema', 'Consolidation']
    pivot_table = pivot_table.set_index('Category Name').reindex(row_order).reset_index()
    
    # Ensure column order with MEAN at the end
    column_order = ['Category Name', 'CX', 'NIH', 'PC', 'MEAN']
    pivot_table = pivot_table[column_order]
    
    # Display with nice formatting
    from IPython.display import display
    styled_table = pivot_table.style.set_properties(**{
        'border': '1px solid black', 
        'text-align': 'center'
    }).set_table_styles([
        {'selector': 'th', 'props': [('border', '1px solid black'), ('text-align', 'center'), ('background-color', '#f2f2f2')]},
        {'selector': 'td', 'props': [('border', '1px solid black'), ('text-align', 'center')]}
    ])
    
    print("\nPerformance Results with Average Thresholds:")
    display(styled_table)
    
    return pivot_table

def plot_pathology_confusion_matrices_comprehensive(cx_data, cx_labels, cx_optimal_thresholds,
                                            nih_data, nih_labels, nih_optimal_thresholds,
                                            pc_data, pc_labels, pc_optimal_thresholds,
                                            results_dir):
    """Create comprehensive confusion matrices for both rejected and non-rejected samples by pathology."""
    
    # Initialize dictionaries to store sample data for each pathology
    pathology_all_labels = {str(cat): [] for cat in CATEGORY_MAPPING.keys()}
    pathology_all_predictions = {str(cat): [] for cat in CATEGORY_MAPPING.keys()}
    pathology_rejection_status = {str(cat): [] for cat in CATEGORY_MAPPING.keys()}  # 1 for rejected, 0 for not rejected
    
    # Process each dataset and collect samples by pathology
    datasets = [
        (cx_data, cx_labels, cx_optimal_thresholds, "CX"),
        (nih_data, nih_labels, nih_optimal_thresholds, "NIH"),
        (pc_data, pc_labels, pc_optimal_thresholds, "PC")
    ]
    
    for data, labels, optimal_thresholds, dataset_name in datasets:
        for _, row in optimal_thresholds.iterrows():
            # Ensure 'Category' is handled as a string for DataFrame access
            col = str(row['Category'])
            median_value = row['Median']
            th = row['Threshold']

            # Calculate absolute deviations
            abs_deviations = np.abs(data[col] - median_value)
            
            # Identify rejected and non-rejected samples
            rejected_mask = abs_deviations <= th
            non_rejected_mask = ~rejected_mask
            
            # Standard prediction threshold for binary classification
            pred_threshold = 0.5
            
            # Collect all samples' true labels
            pathology_all_labels[col].extend(labels[col].tolist())
            
            # Create predictions array - assign actual predictions to non-rejected, and 0 to rejected
            predictions = np.zeros(len(data[col]))
            predictions[non_rejected_mask] = (data[col][non_rejected_mask] >= pred_threshold).astype(int)
            pathology_all_predictions[col].extend(predictions.tolist())
            
            # Keep track of which samples were rejected
            rejection_status = rejected_mask.astype(int)
            pathology_rejection_status[col].extend(rejection_status.tolist())
    
    # Plot comprehensive confusion matrix for each pathology
    for cat in CATEGORY_MAPPING.keys():
        cat_str = str(cat)
        
        # Convert lists to numpy arrays for easier manipulation
        all_labels = np.array(pathology_all_labels[cat_str])
        all_predictions = np.array(pathology_all_predictions[cat_str])
        rejection_status = np.array(pathology_rejection_status[cat_str])
        
        # Split into rejected and non-rejected sets
        rejected_labels = all_labels[rejection_status == 1]
        rejected_predictions = all_predictions[rejection_status == 1]  # These should all be 0
        
        non_rejected_labels = all_labels[rejection_status == 0]
        non_rejected_predictions = all_predictions[rejection_status == 0]
        
        # Compute confusion matrices
        rejected_cm = confusion_matrix(rejected_labels, rejected_predictions, labels=[0, 1])
        non_rejected_cm = confusion_matrix(non_rejected_labels, non_rejected_predictions, labels=[0, 1])
        combined_cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1])
        
        # Calculate metrics
        if len(rejected_labels) > 0:
            rejected_tn, rejected_fp, rejected_fn, rejected_tp = rejected_cm.ravel()
        else:
            rejected_tn, rejected_fp, rejected_fn, rejected_tp = 0, 0, 0, 0
            
        if len(non_rejected_labels) > 0:
            non_rejected_tn, non_rejected_fp, non_rejected_fn, non_rejected_tp = non_rejected_cm.ravel()
        else:
            non_rejected_tn, non_rejected_fp, non_rejected_fn, non_rejected_tp = 0, 0, 0, 0
            
        combined_tn, combined_fp, combined_fn, combined_tp = combined_cm.ravel()
        
        # Create a comprehensive visualization with multiple subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot rejected samples confusion matrix
        sns.heatmap(rejected_cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['True Negative', 'True Positive'], ax=axes[0])
        axes[0].set_title(f'Rejected Samples - {CATEGORY_MAPPING[int(cat)]}')
        axes[0].set_xlabel('Predicted Label')
        axes[0].set_ylabel('Actual Label')
        
        # Plot non-rejected samples confusion matrix
        sns.heatmap(non_rejected_cm, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['True Negative', 'True Positive'], ax=axes[1])
        axes[1].set_title(f'Non-Rejected Samples - {CATEGORY_MAPPING[int(cat)]}')
        axes[1].set_xlabel('Predicted Label')
        axes[1].set_ylabel('Actual Label')
        
        # Plot combined confusion matrix
        sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Purples', 
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['True Negative', 'True Positive'], ax=axes[2])
        axes[2].set_title(f'All Samples - {CATEGORY_MAPPING[int(cat)]}')
        axes[2].set_xlabel('Predicted Label')
        axes[2].set_ylabel('Actual Label')
        
        plt.tight_layout()
        
        # Save plot
        confusion_matrix_path = os.path.join(results_dir, f'pathology_{cat}_comprehensive_confusion_matrix.png')
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created comprehensive confusion matrix for {CATEGORY_MAPPING[int(cat)]}: {confusion_matrix_path}")
        
        # Print detailed statistics
        print(f"\nStatistics for {CATEGORY_MAPPING[int(cat)]}:")
        
        total_samples = len(all_labels)
        rejected_count = len(rejected_labels)
        non_rejected_count = len(non_rejected_labels)
        rejection_rate = rejected_count / total_samples if total_samples > 0 else 0
        
        print(f"  Total samples: {total_samples}")
        print(f"  Rejected samples: {rejected_count} ({rejection_rate:.2%})")
        print(f"  Non-rejected samples: {non_rejected_count} ({1-rejection_rate:.2%})")
        
        # Calculate metrics for rejected samples
        if rejected_count > 0:
            rejected_accuracy = (rejected_tn + rejected_tp) / rejected_count
            rejected_fn_rate = rejected_fn / (rejected_fn + rejected_tp) if (rejected_fn + rejected_tp) > 0 else 0
            print(f"  Rejected samples metrics:")
            print(f"    True Negatives: {rejected_tn}")
            print(f"    False Positives: {rejected_fp}")
            print(f"    False Negatives: {rejected_fn}")
            print(f"    True Positives: {rejected_tp}")
            print(f"    Accuracy: {rejected_accuracy:.4f}")
            print(f"    False Negative Rate: {rejected_fn_rate:.4f}")
        
        # Calculate metrics for non-rejected samples
        if non_rejected_count > 0:
            non_rejected_accuracy = (non_rejected_tn + non_rejected_tp) / non_rejected_count
            non_rejected_precision = non_rejected_tp / (non_rejected_tp + non_rejected_fp) if (non_rejected_tp + non_rejected_fp) > 0 else 0
            non_rejected_recall = non_rejected_tp / (non_rejected_tp + non_rejected_fn) if (non_rejected_tp + non_rejected_fn) > 0 else 0
            non_rejected_f1 = 2 * (non_rejected_precision * non_rejected_recall) / (non_rejected_precision + non_rejected_recall) if (non_rejected_precision + non_rejected_recall) > 0 else 0
            
            print(f"  Non-rejected samples metrics:")
            print(f"    True Negatives: {non_rejected_tn}")
            print(f"    False Positives: {non_rejected_fp}")
            print(f"    False Negatives: {non_rejected_fn}")
            print(f"    True Positives: {non_rejected_tp}")
            print(f"    Accuracy: {non_rejected_accuracy:.4f}")
            print(f"    Precision: {non_rejected_precision:.4f}")
            print(f"    Recall: {non_rejected_recall:.4f}")
            print(f"    F1 Score: {non_rejected_f1:.4f}")
        
        # Calculate metrics for all samples
        if total_samples > 0:
            combined_accuracy = (combined_tn + combined_tp) / total_samples
            combined_precision = combined_tp / (combined_tp + combined_fp) if (combined_tp + combined_fp) > 0 else 0
            combined_recall = combined_tp / (combined_tp + combined_fn) if (combined_tp + combined_fn) > 0 else 0
            combined_f1 = 2 * (combined_precision * combined_recall) / (combined_precision + combined_recall) if (combined_precision + combined_recall) > 0 else 0
            
            print(f"  All samples metrics:")
            print(f"    True Negatives: {combined_tn}")
            print(f"    False Positives: {combined_fp}")
            print(f"    False Negatives: {combined_fn}")
            print(f"    True Positives: {combined_tp}")
            print(f"    Accuracy: {combined_accuracy:.4f}")
            print(f"    Precision: {combined_precision:.4f}")
            print(f"    Recall: {combined_recall:.4f}")
            print(f"    F1 Score: {combined_f1:.4f}")
    
    return results_dir

def main():
    # Base path (modify as needed)
    base_path = r"C:\Users\Ravit\Desktop\מערכות תבוניות\שנה ב\סמסטר א\ראייה ממוחשבת\פרויקט\חלק 2\results"

    # Setup results directory
    results_dir = setup_results_directory(base_path)

    # File paths
    cx_file_train = os.path.join(base_path, 'train_val', 'cx_train.csv')
    nih_file_train = os.path.join(base_path, 'train_val', 'nih_train.csv')
    pc_file_train = os.path.join(base_path, 'train_val', 'pc_train.csv')
    
    cx_file_val = os.path.join(base_path, 'train_val', 'cx_validation.csv')
    nih_file_val = os.path.join(base_path, 'train_val', 'nih_validation.csv')
    pc_file_val = os.path.join(base_path, 'train_val', 'pc_validation.csv')
    
    cx_gt_path_train = os.path.join(base_path, 'train_val', 'cx_gt_train.csv')
    nih_gt_path_train = os.path.join(base_path, 'train_val', 'nih_gt_train.csv')
    pc_gt_path_train = os.path.join(base_path, 'train_val', 'pc_gt_train.csv')

    cx_gt_path_val = os.path.join(base_path, 'train_val', 'cx_gt_validation.csv')
    nih_gt_path_val = os.path.join(base_path, 'train_val', 'nih_gt_validation.csv')
    pc_gt_path_val = os.path.join(base_path, 'train_val', 'pc_gt_validation.csv')

    # Load data
    cx_train = pd.read_csv(cx_file_train)
    nih_train = pd.read_csv(nih_file_train)
    pc_train = pd.read_csv(pc_file_train)
    
    cx_val = pd.read_csv(cx_file_val)
    nih_val = pd.read_csv(nih_file_val)
    pc_val = pd.read_csv(pc_file_val)
    
    cx_gt_train = pd.read_csv(cx_gt_path_train)
    nih_gt_train = pd.read_csv(nih_gt_path_train)
    pc_gt_train = pd.read_csv(pc_gt_path_train)

    cx_gt_val = pd.read_csv(cx_gt_path_val)
    nih_gt_val = pd.read_csv(nih_gt_path_val)
    pc_gt_val = pd.read_csv(pc_gt_path_val)

    # 1. Compute distribution summaries
    cx_distribution = compute_distribution_optimized(cx_train, cx_gt_train, "CX")
    nih_distribution = compute_distribution_optimized(nih_train, nih_gt_train, "NIH")
    pc_distribution = compute_distribution_optimized(pc_train, pc_gt_train, "PC")
    
    print("1. Distribution Summaries:")
    distribution_summary = pd.concat([cx_distribution, nih_distribution, pc_distribution])
    # Display with proper category names
    columns_to_display = ["Dataset","Category Name", "Median", "Std Dev", "Skewness"]
    print(distribution_summary[columns_to_display].to_string(index=False))

    # 2. Find optimal thresholds with parallel processing
    cx_optimal_thresholds = find_optimal_thresholds_parallel(cx_train, cx_gt_train, cx_distribution, 'CX', results_dir)
    nih_optimal_thresholds = find_optimal_thresholds_parallel(nih_train, nih_gt_train, nih_distribution, 'NIH', results_dir)
    pc_optimal_thresholds = find_optimal_thresholds_parallel(pc_train, pc_gt_train, pc_distribution, 'PC', results_dir)

    # Calculate average thresholds for each pathology
    avg_thresholds = calculate_average_thresholds(cx_optimal_thresholds, nih_optimal_thresholds, pc_optimal_thresholds)

    # 3. Apply rejection mechanism with average thresholds
    cx_rejection_performance, cx_predictions, cx_labels = apply_rejection_mechanism_with_avg_thresholds(cx_val, cx_gt_val, avg_thresholds)
    nih_rejection_performance, nih_predictions, nih_labels = apply_rejection_mechanism_with_avg_thresholds(nih_val, nih_gt_val, avg_thresholds)
    pc_rejection_performance, pc_predictions, pc_labels = apply_rejection_mechanism_with_avg_thresholds(pc_val, pc_gt_val, avg_thresholds)

    # Combine all predictions and labels for overall ROC curve
    all_predictions = np.concatenate([cx_predictions, nih_predictions, pc_predictions])
    all_labels = np.concatenate([cx_labels, nih_labels, pc_labels])

    # 4. Plot overall ROC curve
    plot_overall_roc_curve(all_predictions, all_labels, results_dir)

    # Keep the confusion matrix plots
    plot_pathology_confusion_matrices_comprehensive(
    cx_val, cx_gt_val, cx_optimal_thresholds,
    nih_val, nih_gt_val, nih_optimal_thresholds,
    pc_val, pc_gt_val, pc_optimal_thresholds,
    results_dir)
    
    # 5. Save rejection performance results
    performance_table = display_rejection_performance(cx_rejection_performance, nih_rejection_performance, pc_rejection_performance)

if __name__ == "__main__":
    main()


# In[ ]:




