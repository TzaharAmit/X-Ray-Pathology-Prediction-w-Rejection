import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix
import seaborn as sns

# ================================
# Part 1: Training Set Analysis with Percentile Optimization
# ================================

# Define datasets and pathologies
datasets = ['pc', 'nih', 'cx']
pathologies = ["Cardiomegaly", "Effusion", "Edema", "Consolidation"]

# To store overall results for different percentiles (Training)
results = []

# Dictionary to store the final rejection mask for each dataset
final_reject_mask_dict = {}

# Dictionary to store the final rejection mask for each dataset (validation)
final_reject_mask_val_dict = {}

# Define the range of percentiles to test (from 75% to 95% in steps of 2%)
percentiles = list(range(75, 96, 2))  # e.g., 75,77,79,...,95

# Process each dataset on the training set
for dataset in datasets:
    # Load model outputs and ground truth CSV files for training
    task_outputs = pd.read_csv(
        rf"C:\Users\talve\OneDrive\שולחן העבודה\פרויקט ראיה ממוחשבת\תוצאות עמית\{dataset}\{dataset}_train.csv")
    task_targets = pd.read_csv(
        rf"C:\Users\talve\OneDrive\שולחן העבודה\פרויקט ראיה ממוחשבת\תוצאות עמית\{dataset}\{dataset}_gt_train.csv")

    print(f"Training Dataset: {dataset}")
    print("Min Confidence per Pathology:")
    print(task_outputs.min(axis=0))
    print("Max Confidence per Pathology:")
    print(task_outputs.max(axis=0))

    num_images = len(task_outputs)

    # Loop over percentiles to set adaptive entropy thresholds
    for perc in percentiles:
        # Compute per-pathology adaptive entropy thresholds on training set
        adaptive_entropy_thresholds = {}
        for i, pathology in enumerate(pathologies):
            scores = task_outputs.iloc[:, i].values
            labels = task_targets.iloc[:, i].values

            # Binary predictions at 0.5 cutoff
            preds = (scores >= 0.5).astype(int)
            mask_correct = (preds == labels)

            # Compute binary entropy for each sample
            p = np.clip(scores, 1e-10, 1.0)
            entropy_vals = - (p * np.log(p) + (1 - p) * np.log(1 - p))

            if np.sum(mask_correct) > 0:
                threshold = np.percentile(entropy_vals[mask_correct], perc)
            else:
                threshold = np.nan
            adaptive_entropy_thresholds[pathology] = threshold

        # Create image-level rejection mask (reject if ALL pathology entropies exceed thresholds)
        reject_image_mask = np.zeros(num_images, dtype=bool)
        for idx in range(num_images):
            all_above = True
            for i, pathology in enumerate(pathologies):
                score = task_outputs.iloc[idx, i]
                p_val = np.clip(score, 1e-10, 1.0)
                ent = - (p_val * np.log(p_val) + (1 - p_val) * np.log(1 - p_val))
                if not np.isnan(adaptive_entropy_thresholds[pathology]):
                    if ent <= adaptive_entropy_thresholds[pathology]:
                        all_above = False
                        break
                else:
                    all_above = False
                    break
            if all_above:
                reject_image_mask[idx] = True

        # For each pathology, compute metrics on accepted images
        for i, pathology in enumerate(pathologies):
            scores = task_outputs.iloc[:, i].values
            labels = task_targets.iloc[:, i].values

            try:
                auc_before = roc_auc_score(labels, scores)
            except Exception:
                auc_before = np.nan
            preds_before = (scores >= 0.5).astype(int)
            f1_before = f1_score(labels, preds_before)

            accepted_mask = ~reject_image_mask
            accepted_scores = scores[accepted_mask]
            accepted_labels = labels[accepted_mask]

            if len(accepted_scores) > 0:
                try:
                    auc_after = roc_auc_score(accepted_labels, accepted_scores)
                except Exception:
                    auc_after = np.nan
                preds_after = (accepted_scores >= 0.5).astype(int)
                f1_after = f1_score(accepted_labels, preds_after)
            else:
                auc_after = np.nan
                f1_after = np.nan

            pathology_present = (labels == 1)
            if np.sum(pathology_present) > 0:
                rejection_rate = np.sum(reject_image_mask & pathology_present) / np.sum(pathology_present) * 100
            else:
                rejection_rate = 0

            results.append({
                'Dataset': dataset,
                'Pathology': pathology,
                'Percentile': perc,
                'Entropy Threshold': round(adaptive_entropy_thresholds[pathology], 4) if not np.isnan(
                    adaptive_entropy_thresholds[pathology]) else "N/A",
                'AUC Before': round(auc_before * 100, 2) if not np.isnan(auc_before) else "N/A",
                'AUC After': round(auc_after * 100, 2) if not np.isnan(auc_after) else "N/A",
                'F1 Before (%)': round(f1_before * 100, 2),
                'F1 After (%)': round(f1_after * 100, 2) if not np.isnan(f1_after) else "N/A",
                'Rejection Rate (%)': round(rejection_rate, 2)
            })
    # End of percentile loop for current dataset.
    # Store the final rejection mask for this dataset (from the last percentile iteration)
    final_reject_mask_dict[dataset] = reject_image_mask.copy()

# Convert training results to a DataFrame and save as CSV
df_results = pd.DataFrame(results)
output_csv = r"C:\Users\talve\OneDrive\שולחן העבודה\פרויקט ראיה ממוחשבת\results\Rejection8\image_level_rejection_metrics_percentile.csv"
df_results.to_csv(output_csv, index=False)
print(df_results)

# Make sure "AUC After" and "Percentile" are numeric
df_results['AUC After'] = pd.to_numeric(df_results['AUC After'], errors='coerce')
df_results['Percentile'] = pd.to_numeric(df_results['Percentile'], errors='coerce')

# 1) For each (Dataset, Pathology), find the row with the maximum "AUC After"
optimal_percentiles = df_results.groupby(['Dataset', 'Pathology']).apply(
    lambda grp: grp.loc[grp['AUC After'].idxmax(), ['Percentile','AUC After']]
).reset_index()
print("Optimal Percentile per (Dataset, Pathology):")
print(optimal_percentiles)

# 2) If you want a single percentile per pathology (averaged across all datasets):
avg_optimal_df = optimal_percentiles.groupby('Pathology')['Percentile'].mean().reset_index()
avg_optimal_df.rename(columns={'Percentile': 'Average Optimal Percentile'}, inplace=True)
print("Average Optimal Percentile per Pathology:")
print(avg_optimal_df)

# 3) Write that out to CSV so Part 2 can load it
avg_optimal_csv = r"C:\Users\talve\OneDrive\שולחן העבודה\פרויקט ראיה ממוחשבת\results\Rejection8\avg_optimal_percentile.csv"
avg_optimal_df.to_csv(avg_optimal_csv, index=False)
print(f"Saved average optimal percentiles to {avg_optimal_csv}")


# Visualization for Training: One plot per dataset (AUC After vs. Rejection Rate)
for dataset in datasets:
    plt.figure(figsize=(10, 6))
    for pathology in pathologies:
        df_plot = df_results[(df_results['Dataset'] == dataset) & (df_results['Pathology'] == pathology)]
        x = df_plot['Rejection Rate (%)'].astype(float)
        y = df_plot['AUC After'].astype(float)
        plt.plot(x, y, marker='o', linestyle='-', label=pathology)
        for idx, row in df_plot.iterrows():
            plt.annotate(f"{row['Percentile']}%", (row['Rejection Rate (%)'], row['AUC After']),
                         textcoords="offset points", xytext=(5, 5), fontsize=8)
        if not df_plot.empty:
            df_plot_numeric = df_plot.copy()
            df_plot_numeric['AUC After'] = df_plot_numeric['AUC After'].astype(float)
            idx_max = df_plot_numeric['AUC After'].idxmax()
            max_row = df_plot_numeric.loc[idx_max]
            plt.scatter(max_row['Rejection Rate (%)'], max_row['AUC After'], color='red', zorder=5)
            plt.annotate(f"Max {max_row['Percentile']}%",
                         (max_row['Rejection Rate (%)'], max_row['AUC After']),
                         textcoords="offset points", xytext=(5, -10), fontsize=10, color='red')
    plt.xlabel("Rejection Rate (%)")
    plt.ylabel("AUC After (%)")
    plt.title(f"{dataset}: AUC After vs. Rejection Rate (Percentile Analysis)")
    plt.grid(True)
    plt.legend(title="Pathology")
    plt.tight_layout()
    plt.savefig(
        rf"C:\Users\talve\OneDrive\שולחן העבודה\פרויקט ראיה ממוחשבת\results\Rejection8\{dataset}_auc_vs_rejection.png")
    plt.close()

# ================================
# Part 2: Validation Set Analysis using Average Optimal Percentile Values
# ================================

# Load the average optimal percentile values from CSV
avg_optimal_df = pd.read_csv(
    r"C:\Users\talve\OneDrive\שולחן העבודה\פרויקט ראיה ממוחשבת\results\Rejection8\avg_optimal_percentile.csv")
avg_optimal_dict = avg_optimal_df.set_index('Pathology')['Average Optimal Percentile'].to_dict()
print("Average Optimal Percentile Dictionary (Validation):", avg_optimal_dict)

validation_results = []

for dataset in datasets:
    val_outputs = pd.read_csv(
        rf"C:\Users\talve\OneDrive\שולחן העבודה\פרויקט ראיה ממוחשבת\תוצאות עמית\{dataset}\{dataset}_validation.csv")
    val_targets = pd.read_csv(
        rf"C:\Users\talve\OneDrive\שולחן העבודה\פרויקט ראיה ממוחשבת\תוצאות עמית\{dataset}\{dataset}_gt_validation.csv")

    print(f"Validation Dataset: {dataset}")
    num_images = len(val_outputs)

    adaptive_entropy_thresholds_val = {}
    for i, pathology in enumerate(pathologies):
        scores = val_outputs.iloc[:, i].values
        labels = val_targets.iloc[:, i].values
        preds = (scores >= 0.5).astype(int)
        mask_correct = (preds == labels)
        p = np.clip(scores, 1e-10, 1.0)
        entropy_vals = - (p * np.log(p) + (1 - p) * np.log(1 - p))

        avg_perc = avg_optimal_dict.get(pathology, 90)
        if np.sum(mask_correct) > 0:
            threshold = np.percentile(entropy_vals[mask_correct], avg_perc)
        else:
            threshold = np.nan
        adaptive_entropy_thresholds_val[pathology] = threshold
        print(
            f"Validation - Dataset: {dataset}, Pathology: {pathology}, Avg Optimal Percentile: {avg_perc}, Threshold: {threshold}")

    reject_image_mask_val = np.zeros(num_images, dtype=bool)
    for idx in range(num_images):
        all_above = True
        for i, pathology in enumerate(pathologies):
            score = val_outputs.iloc[idx, i]
            p = np.clip(score, 1e-10, 1.0)
            ent = - (p * np.log(p) + (1 - p) * np.log(1 - p))
            if not np.isnan(adaptive_entropy_thresholds_val[pathology]):
                if ent <= adaptive_entropy_thresholds_val[pathology]:
                    all_above = False
                    break
            else:
                all_above = False
                break
        if all_above:
            reject_image_mask_val[idx] = True

        # store the rejection mask for this dataset so we can do confusion matrices later
        final_reject_mask_val_dict[dataset] = reject_image_mask_val.copy()


    for i, pathology in enumerate(pathologies):
        scores = val_outputs.iloc[:, i].values
        labels = val_targets.iloc[:, i].values

        try:
            auc_before = roc_auc_score(labels, scores)
        except Exception:
            auc_before = np.nan
        preds_before = (scores >= 0.5).astype(int)
        f1_before = f1_score(labels, preds_before)

        accepted_mask = ~reject_image_mask_val
        accepted_scores = scores[accepted_mask]
        accepted_labels = labels[accepted_mask]

        if len(accepted_scores) > 0:
            try:
                auc_after = roc_auc_score(accepted_labels, accepted_scores)
            except Exception:
                auc_after = np.nan
            preds_after = (accepted_scores >= 0.5).astype(int)
            f1_after = f1_score(accepted_labels, preds_after)
        else:
            auc_after = np.nan
            f1_after = np.nan

        pathology_present = (labels == 1)
        if np.sum(pathology_present) > 0:
            rejection_rate = np.sum(reject_image_mask_val & pathology_present) / np.sum(pathology_present) * 100
        else:
            rejection_rate = 0

        validation_results.append({
            'Dataset': dataset,
            'Pathology': pathology,
            'Avg Optimal Percentile': avg_perc,
            'Entropy Threshold': round(adaptive_entropy_thresholds_val[pathology], 4) if not np.isnan(
                adaptive_entropy_thresholds_val[pathology]) else "N/A",
            'AUC Before': round(auc_before * 100, 2) if not np.isnan(auc_before) else "N/A",
            'AUC After': round(auc_after * 100, 2) if not np.isnan(auc_after) else "N/A",
            'F1 Before (%)': round(f1_before * 100, 2),
            'F1 After (%)': round(f1_after * 100, 2) if not np.isnan(f1_after) else "N/A",
            'Rejection Rate (%)': round(rejection_rate, 2)
        })

# Save validation results to CSV
df_val_results = pd.DataFrame(validation_results)
val_output_csv = r"C:\Users\talve\OneDrive\שולחן העבודה\פרויקט ראיה ממוחשבת\results\Rejection8\image_level_rejection_metrics_validation.csv"
df_val_results.to_csv(val_output_csv, index=False)
print("Validation Results:")
print(df_val_results)

# ================================
# Part 3: Additional Analysis - Confusion Matrices for Rejected Cases
# (For each dataset and pathology, using the rejection mask from the training set stored in final_reject_mask_dict.)
for dataset in datasets:
    # Load the training files (should be same as used above)
    val_outputs = pd.read_csv(
        rf"C:\Users\talve\OneDrive\שולחן העבודה\פרויקט ראיה ממוחשבת\תוצאות עמית\{dataset}\{dataset}_validation.csv")
    val_targets = pd.read_csv(
        rf"C:\Users\talve\OneDrive\שולחן העבודה\פרויקט ראיה ממוחשבת\תוצאות עמית\{dataset}\{dataset}_gt_validation.csv")

    # Retrieve the stored rejection mask for this dataset from training analysis
    #reject_image_mask = final_reject_mask_dict[dataset]  # This mask has the correct length for this dataset.

    # Retrieve the stored mask for the validation set
    rejected_mask_val = final_reject_mask_val_dict[dataset]

    for pathology in pathologies:
        col_idx = pathologies.index(pathology)
        scores = val_outputs.iloc[:, col_idx].values
        labels = val_targets.iloc[:, col_idx].values

        # Filter out the rejected samples
        rejected_scores = scores[rejected_mask_val]
        rejected_labels = labels[rejected_mask_val]

        if len(rejected_scores) == 0:
            print(f"No rejected samples for {dataset} - {pathology}. Skipping confusion matrix.")
            continue

        # Predictions at 0.5 cutoff for the rejected samples
        rejected_preds = (rejected_scores >= 0.5).astype(int)

        # Compute confusion matrix
        cm_rejected = confusion_matrix(rejected_labels, rejected_preds)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_rejected, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
        plt.title(f"Rejected Cases Confusion Matrix\nDataset: {dataset} - {pathology}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(
            rf"C:\Users\talve\OneDrive\שולחן העבודה\פרויקט ראיה ממוחשבת\results\Rejection8\{dataset}_{pathology}_confusion_rejected_val.png")
        plt.close()