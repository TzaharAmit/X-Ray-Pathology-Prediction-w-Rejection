import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import seaborn as sns

# Define all dataset names and path for each folder (adjust if needed)
datasets = ['pc', 'nih', 'cx']
data_base_path = r"C:\Users\talve\OneDrive\שולחן העבודה\פרויקט ראיה ממוחשבת"

# Filenames in each folder
outputs_filename = "task_outputs.csv"
targets_filename = "task_targets.csv"

# Define pathologies (assuming same column order in each CSV)
pathologies = ["Cardiomegaly", "Effusion", "Edema", "Consolidation"]

# Define range of percentiles to test (e.g., 75% to 95% in steps of 2%)
percentiles = list(range(75, 100, 2))

# We'll store training results for each combination to later derive optimal percentiles.
all_train_results = []  # list of dictionaries
# Also store the thresholds per combination for later use.
train_thresholds_dict = {}  # key: (train_datasets_tuple), value: dict{pathology: optimal threshold}

# ------------------------------
# Part 1: Training Analysis on Merged Train Sets (2 datasets)
# ------------------------------
# Generate combinations: each combination is (train_set, validation_set)
# For 3 datasets, the combinations are:
#    Train: [pc, nih], Validation: cx
#    Train: [pc, cx], Validation: nih
#    Train: [nih, cx], Validation: pc
combos = list(combinations(datasets, 2))
print("Train combinations:", combos)

# For each combination, merge the two training sets
for train_combo in combos:
    # The validation set is the one not in the training combination.
    val_set = list(set(datasets) - set(train_combo))[0]
    print(f"\nTrain: {train_combo} | Validation: {val_set}")

    # Load and merge training outputs and targets from the two training datasets.
    train_outputs_list = []
    train_targets_list = []
    for d in train_combo:
        out_path = os.path.join(data_base_path, d, outputs_filename)
        tar_path = os.path.join(data_base_path, d, targets_filename)
        train_outputs_list.append(pd.read_csv(out_path))
        train_targets_list.append(pd.read_csv(tar_path))

    train_outputs = pd.concat(train_outputs_list, axis=0, ignore_index=True)
    train_targets = pd.concat(train_targets_list, axis=0, ignore_index=True)
    num_train = len(train_outputs)

    # For each tested percentile, compute adaptive entropy thresholds on the merged training set
    for perc in percentiles:
        adaptive_entropy_thresholds = {}
        # For each pathology, compute threshold using correctly classified samples
        for i, pathology in enumerate(pathologies):
            scores = train_outputs.iloc[:, i].values
            labels = train_targets.iloc[:, i].values

            # Predictions at 0.5 cutoff and correct sample mask
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

        # Create image-level rejection mask on the merged training set
        reject_mask = np.zeros(num_train, dtype=bool)
        for idx in range(num_train):
            all_above = True
            for i, pathology in enumerate(pathologies):
                score = train_outputs.iloc[idx, i]
                p_val = np.clip(score, 1e-10, 1.0)
                ent = - (p_val * np.log(p_val) + (1 - p_val) * np.log(1 - p_val))
                # Reject if for this pathology, entropy > threshold.
                if not np.isnan(adaptive_entropy_thresholds[pathology]):
                    if ent <= adaptive_entropy_thresholds[pathology]:
                        all_above = False
                        break
                else:
                    all_above = False
                    break
            if all_above:
                reject_mask[idx] = True

        # For each pathology, compute metrics on accepted training samples
        for i, pathology in enumerate(pathologies):
            scores = train_outputs.iloc[:, i].values
            labels = train_targets.iloc[:, i].values

            try:
                auc_before = roc_auc_score(labels, scores)
            except Exception:
                auc_before = np.nan
            preds_before = (scores >= 0.5).astype(int)
            f1_before = f1_score(labels, preds_before)

            accepted_mask = ~reject_mask
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
                rejection_rate = np.sum(reject_mask & pathology_present) / np.sum(pathology_present) * 100
            else:
                rejection_rate = 0

            # --- Limit rejection rate to maximum 25% ---
            if rejection_rate <= 25:
                all_train_results.append({
                    'TrainCombo': str(train_combo),
                    'Validation': val_set,
                    'Percentile': perc,
                    'Pathology': pathology,
                    'Entropy Threshold': round(adaptive_entropy_thresholds[pathology], 4) if not np.isnan(
                        adaptive_entropy_thresholds[pathology]) else "N/A",
                    'AUC Before': round(auc_before * 100, 2) if not np.isnan(auc_before) else "N/A",
                    'AUC After': round(auc_after * 100, 2) if not np.isnan(auc_after) else "N/A",
                    'F1 Before (%)': round(f1_before * 100, 2),
                    'F1 After (%)': round(f1_after * 100, 2) if not np.isnan(f1_after) else "N/A",
                    'Rejection Rate (%)': round(rejection_rate, 2)
                })

        # (Optional: You can store thresholds per percentile if needed.)

    # For this training combo, you might want to determine the optimal percentile per pathology.
    # For simplicity, here we only save the results.

# Save training results for merged train sets to CSV
df_train_results = pd.DataFrame(all_train_results)
train_output_csv = os.path.join(data_base_path, "results", "Rejection9", "merged_train_results.csv")
os.makedirs(os.path.dirname(train_output_csv), exist_ok=True)
df_train_results.to_csv(train_output_csv, index=False)
print(df_train_results)

# ------------------------------
# Part 2: Validation Set Analysis using the Optimal Thresholds from Training
# ------------------------------
# For each combination, use the training set optimal thresholds (for each pathology) to process the validation set.
# For simplicity, assume that we determine optimal percentile per pathology by choosing the one with maximum AUC After
# from df_train_results, then compute the average across combinations.
optimal_percentiles = df_train_results.groupby(['TrainCombo', 'Pathology']).apply(
    lambda grp: grp.loc[grp['AUC After'].idxmax(), ['Percentile', 'AUC After']]
).reset_index()
print("Optimal Percentile per TrainCombo and Pathology:")
print(optimal_percentiles)

# Now, average the optimal percentiles per pathology across all train combinations.
avg_optimal = optimal_percentiles.groupby('Pathology')['Percentile'].mean().reset_index()
avg_optimal.rename(columns={'Percentile': 'Average Optimal Percentile'}, inplace=True)
print("Average Optimal Percentile per Pathology (across train combos):")
print(avg_optimal)

# Save the avg optimal percentile to CSV
avg_optimal_csv = os.path.join(data_base_path, "results", "Rejection9", "avg_optimal_percentile.csv")
avg_optimal.to_csv(avg_optimal_csv, index=False)

# Create a dictionary for optimal percentiles per pathology.
avg_optimal_dict = avg_optimal.set_index('Pathology')['Average Optimal Percentile'].to_dict()

# Now, for each train combo, use its corresponding validation set to compute metrics.
# We'll loop over the same train combos we used.
validation_results = []
# Let's recompute the combinations
for train_combo in combos:
    val_set = list(set(datasets) - set(train_combo))[0]
    print(f"\nFor Train: {train_combo} | Validation: {val_set}")

    # Load validation files for the held-out dataset
    val_outputs = pd.read_csv(
        os.path.join(data_base_path, val_set, outputs_filename))
    val_targets = pd.read_csv(
        os.path.join(data_base_path, val_set, targets_filename))

    num_val = len(val_outputs)

    # Compute adaptive entropy thresholds on the validation set using the average optimal percentiles
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
        print(f"Validation - {val_set}, {pathology}: Avg Optimal Percentile = {avg_perc}, Threshold = {threshold}")

    # Create an image-level rejection mask on the validation set.
    reject_mask_val = np.zeros(num_val, dtype=bool)
    for idx in range(num_val):
        all_above = True
        for i, pathology in enumerate(pathologies):
            score = val_outputs.iloc[idx, i]
            p_val = np.clip(score, 1e-10, 1.0)
            ent = - (p_val * np.log(p_val) + (1 - p_val) * np.log(1 - p_val))
            if not np.isnan(adaptive_entropy_thresholds_val[pathology]):
                if ent <= adaptive_entropy_thresholds_val[pathology]:
                    all_above = False
                    break
            else:
                all_above = False
                break
        if all_above:
            reject_mask_val[idx] = True

    # Compute metrics for each pathology on accepted validation images.
    for i, pathology in enumerate(pathologies):
        scores = val_outputs.iloc[:, i].values
        labels = val_targets.iloc[:, i].values

        try:
            auc_before = roc_auc_score(labels, scores)
        except Exception:
            auc_before = np.nan
        preds_before = (scores >= 0.5).astype(int)
        f1_before = f1_score(labels, preds_before)

        accepted_mask = ~reject_mask_val
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
            rejection_rate = np.sum(reject_mask_val & pathology_present) / np.sum(pathology_present) * 100
        else:
            rejection_rate = 0

        validation_results.append({
            'TrainCombo': str(train_combo),
            'Validation': val_set,
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
val_output_csv = os.path.join(data_base_path, "results", "Rejection9", "merged_validation_results.csv")
os.makedirs(os.path.dirname(val_output_csv), exist_ok=True)
df_val_results.to_csv(val_output_csv, index=False)
print("Merged Validation Results:")
print(df_val_results)

# ================================
# Part 3: (Optional) Confusion Matrix for Rejected Cases on Validation Set
# ================================
# For each combination, you can also compute and plot confusion matrices for the rejected validation samples.
for combo in combos:
    val_set = list(set(datasets) - set(combo))[0]
    # Load validation data for the held-out set
    val_outputs = pd.read_csv(
        os.path.join(data_base_path, val_set, outputs_filename))
    val_targets = pd.read_csv(
        os.path.join(data_base_path, val_set, targets_filename))
    num_val = len(val_outputs)

    # For the current combo, assume the adaptive thresholds computed above are used (they come from the merged training set).
    # We'll re-compute them for each pathology on the validation set using avg_optimal_dict.
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

    # Create a rejection mask on the validation set
    reject_mask_val = np.zeros(num_val, dtype=bool)
    for idx in range(num_val):
        all_above = True
        for i, pathology in enumerate(pathologies):
            score = val_outputs.iloc[idx, i]
            p_val = np.clip(score, 1e-10, 1.0)
            ent = - (p_val * np.log(p_val) + (1 - p_val) * np.log(1 - p_val))
            if not np.isnan(adaptive_entropy_thresholds_val[pathology]):
                if ent <= adaptive_entropy_thresholds_val[pathology]:
                    all_above = False
                    break
            else:
                all_above = False
                break
        if all_above:
            reject_mask_val[idx] = True

    # For each pathology, plot confusion matrix for the rejected validation samples
    for pathology in pathologies:
        col_idx = pathologies.index(pathology)
        scores = val_outputs.iloc[:, col_idx].values
        labels = val_targets.iloc[:, col_idx].values
        rejected_scores = scores[reject_mask_val]
        rejected_labels = labels[reject_mask_val]
        if len(rejected_scores) == 0:
            print(f"No rejected samples for TrainCombo: {combo}, Validation: {val_set}, Pathology: {pathology}")
            continue
        rejected_preds = (rejected_scores >= 0.5).astype(int)
        cm = confusion_matrix(rejected_labels, rejected_preds)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
        plt.title(f"Validation Rejected Cases\nTrain: {combo}, Val: {val_set}, {pathology}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        cm_filename = os.path.join(data_base_path, "results", "Rejection9",
                                   f"{val_set}_{pathology}_confusion_rejected_val.png")
        plt.savefig(cm_filename)
        plt.close()
