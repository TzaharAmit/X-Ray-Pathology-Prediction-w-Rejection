from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, roc_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# List of dataset names
datasets = ['pc', 'nih', 'cx']

# Dictionary to store all metrics for all datasets
all_metrics = {dataset: {} for dataset in datasets}

# Pathologies
pathologies = ["Cardiomegaly", "Effusion", "Edema", "Consolidation"]

# Loop over each dataset
for dataset in datasets:
    # Load task_outputs and task_targets for each dataset
    task_outputs = pd.read_csv(fr"C:\Users\AmitTzahar\OneDrive - Tyto Care Ltd\Documents\afeka\Copmuter Vision\project\results_train_validation\{dataset}\{dataset}_validation.csv")
    task_targets = pd.read_csv(fr"C:\Users\AmitTzahar\OneDrive - Tyto Care Ltd\Documents\afeka\Copmuter Vision\project\results_train_validation\{dataset}\{dataset}_gt_validation.csv")
    
    # Initialize dictionary to store metrics for the current dataset
    dataset_metrics = {}

    # Calculate AUC for each pathology
    auc_scores = {}
    for i, pathology in enumerate(pathologies):
        auc = roc_auc_score(task_targets.iloc[:, i], task_outputs.iloc[:, i])
        auc_scores[pathology] = round(auc * 100, 2)  # Convert to percentage

    # Store AUC scores in the dataset's metrics
    dataset_metrics['AUC'] = auc_scores

    # Convert probabilities to binary predictions (using a threshold of 0.5)
    threshold = 0.5
    binary_outputs = {
        pathology: (task_outputs.iloc[:, i] >= threshold).astype(int)
        for i, pathology in enumerate(pathologies)
    }

    # Calculate accuracy, precision, recall, and F1 score for each pathology
    metrics = {}
    for i, pathology in enumerate(pathologies):
        acc = accuracy_score(task_targets.iloc[:, i], binary_outputs[pathology])
        precision, recall, f1, _ = precision_recall_fscore_support(
            task_targets.iloc[:, i], binary_outputs[pathology], average='binary')
        
        metrics[pathology] = {
            'accuracy': round(acc * 100, 2),       # Convert to percentage
            'precision': round(precision * 100, 2), # Convert to percentage
            'recall': round(recall * 100, 2),       # Convert to percentage
            'f1': round(f1 * 100, 2)                # Convert to percentage
        }
    
    # Store metrics in the dataset's metrics
    dataset_metrics['Metrics'] = metrics

    # Store dataset metrics in all_metrics
    all_metrics[dataset] = dataset_metrics

    # Optional: Print metrics for each dataset (uncomment if needed)
    print(f"Metrics for {dataset}:")
    print(dataset_metrics)

    # Plot ROC curve for each pathology in the current dataset
    plt.figure(figsize=(10, 8))

    for i, pathology in enumerate(pathologies):
        fpr, tpr, thresholds = roc_curve(task_targets.iloc[:, i], task_outputs.iloc[:, i])
        plt.plot(fpr, tpr, label=f'{pathology} (AUC = {auc_scores[pathology]:.2f}%)')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # diagonal line (random classifier)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {dataset} Dataset')
    plt.legend(loc='lower right')

    # Save the ROC plot to a file
    plt.savefig(fr"C:\Users\AmitTzahar\OneDrive - Tyto Care Ltd\Documents\afeka\Copmuter Vision\project\{dataset}_roc_curve.png")
    plt.close()  # Close the plot after saving it to prevent overlap with next dataset

# Convert the results to a DataFrame for CSV export
# Flatten the dictionary into a DataFrame for easy export

rows = []

# Loop over datasets and pathologies to flatten the structure
for dataset in datasets:
    for pathology in pathologies:
        row = {
            'Dataset': dataset,
            'Pathology': pathology,
            'AUC': all_metrics[dataset]['AUC'][pathology],
            'Accuracy': all_metrics[dataset]['Metrics'][pathology]['accuracy'],
            'Precision': all_metrics[dataset]['Metrics'][pathology]['precision'],
            'Recall': all_metrics[dataset]['Metrics'][pathology]['recall'],
            'F1 Score': all_metrics[dataset]['Metrics'][pathology]['f1']
        }
        rows.append(row)

# Create DataFrame from the rows
df_all_metrics = pd.DataFrame(rows)

# Save DataFrame to CSV
df_all_metrics.to_csv(fr"C:\Users\AmitTzahar\OneDrive - Tyto Care Ltd\Documents\afeka\Copmuter Vision\project\all_metrics.csv", index=False)

# Optional: Display DataFrame for verification
print(df_all_metrics)


# AUC metric summary:
df_auc = df_all_metrics[['Dataset', 'Pathology', 'AUC']]
auc_pivot = pd.pivot_table(df_auc, values='AUC', index='Pathology', columns='Dataset')
auc_pivot.loc['dataset_mean_AUC'] = auc_pivot.mean(axis=0)
auc_pivot['pathology_mean_AUC'] = auc_pivot.mean(axis=1)
auc_pivot.to_csv(fr"C:\Users\AmitTzahar\OneDrive - Tyto Care Ltd\Documents\afeka\Copmuter Vision\project\auc_resuls.csv")

# Optional: Display DataFrame for verification
print(auc_pivot)


