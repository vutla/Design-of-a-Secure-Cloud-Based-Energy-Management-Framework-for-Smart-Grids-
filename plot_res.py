import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sns
from load_save import save, load

def plot_res():
    metrics_names = [
        "Accuracy", "Precision", "Sensitivity", "Specificity", "F_measure",
        "MCC", "NPV", "FPR", "FNR", "Jaccard Index",
        "Inference Time", "GMean", "Cross-Entropy Loss", "Cohen's Kappa"
    ]

    models = ["Fed-SCR", "Proposed", "LSTM-MPC", 'ADLA-FL', "Res-block+CNN"]

    values = load('Existing_model_met')

    save_folder = "Result"
    os.makedirs(save_folder, exist_ok=True)
    df = pd.DataFrame(values, columns=metrics_names, index=models)
    excel_path = os.path.join(save_folder, "Model_Metrics.xlsx")
    df.to_excel(excel_path)

    fill_colors = ['#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4',
                   '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4']

    num_metrics = len(metrics_names)
    angles = np.linspace(0, 2 * np.pi, len(models), endpoint=False).tolist()

    for i in range(num_metrics):
        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)

        # Extract values for the current metric
        metric_values = [values[j, i] for j in range(len(models))]

        # Close the radar loop
        metric_values += metric_values[:1]
        angle_values = angles + angles[:1]

        # Plot with different fill color for each metric
        ax.plot(angle_values, metric_values, color='b', linewidth=2)
        ax.fill(angle_values, metric_values, color=fill_colors[i], alpha=0.7)

        # Add value annotations near each point
        for j, angle in enumerate(angles):
            ax.text(angle, metric_values[j], f"{metric_values[j]:.3f}",
                    size=12, color="black", ha='center', va='center', fontweight='bold')

        # Set labels
        ax.set_xticks(angles)
        ax.set_xticklabels(models, fontsize=16, fontweight='bold', rotation=0, ha='center', va='center')

        # Move labels outward using tick_params
        ax.tick_params(axis='x', pad=30)

        ax.set_title(metrics_names[i], size=18, pad=25, fontweight='bold')
        ax.set_yticklabels([])
        ax.grid(True)

        # Save each radar chart as PNG
        filename = f"{metrics_names[i].replace(' ', '_')}.png"
        filepath = os.path.join(save_folder, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=1300, bbox_inches='tight')
        plt.close()


def ROC_AUC():
    def load_and_plot_roc(file_path, output_path):
        # Load AUC, TPR, and FPR values from the pickle file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        auc_values = data['auc']
        tpr_values = data['tpr']
        fpr_values = data['fpr']

        # Plotting the ROC curves
        plt.figure(figsize=(10, 8))

        for classifier in auc_values.keys():
            fpr = fpr_values[classifier]
            tpr = tpr_values[classifier]
            plt.plot([0, fpr, 1], [0, tpr, 1], label=f'{classifier} (AUC = {auc_values[classifier]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
        plt.legend(loc='lower right')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=16, loc='lower right', frameon=True)
        plt.savefig(output_path, dpi=1900)
        plt.show()

        # Print AUC values
        for classifier, auc_value in auc_values.items():
            print(f'{classifier} AUC: {auc_value:.2f}')

    load_and_plot_roc('Saved data/auc_tpr_fpr_values.pkl', 'Result/ROC_Curves.png')


def confusion_met():
    cm = load('cm_metrics')
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["0", "1"],
                yticklabels=["0", "1"])

    plt.title("Confusion Matrix")
    plt.ylabel("True Label", fontsize=15)
    plt.xlabel("Predicted Label", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig("Result/Confusion Matrix.png", dpi=1300)
    plt.show()

    # Classifiers
    classifiers = ["Fed-SCR", "LSTM-MPC", 'ADLA-FL', "Res-block+CNN", "Proposed"]

    # Computation times in seconds (example values)
    comp_time = [323, 201, 256, 145, 110]

    plt.figure(figsize=(12, 6))
    plt.plot(classifiers, comp_time, marker='o', linestyle='-', color='blue', linewidth=2)

    plt.xlabel("Classifier", fontsize=14)
    plt.ylabel("Computation Time (s)", fontsize=14)
    plt.title("Classifier Computation Time", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # Annotate each point with its value
    for i, time in enumerate(comp_time):
        plt.text(i, time + 5, str(time), ha='center', fontsize=14)

    plt.tight_layout()
    plt.savefig('./Result/comptational.png', dpi=1700)
    plt.show()

    classifiers = ["Fed-SCR", "LSTM-MPC", "ADLA-FL", "Res-block+CNN", "Proposed"]
    gflops = [15.6, 14.2, 11.8, 22.4, 6.4]
    inference_speed = [28, 32, 20, 40, 12]

    y = np.arange(len(classifiers))

    # Color sets
    colors1 = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    colors2 = ["#64B5CD", "#E17C05", "#2E8B57", "#B22222", "#8B008B"]
    plt.figure(figsize=(10, 5))
    plt.barh(y, gflops, color=colors1)
    plt.ylabel("Models", fontsize=13)
    plt.xlabel("Computational Resources (GFLOPs)", fontsize=13)
    plt.title("Computational Resources Comparison", fontsize=14)
    plt.yticks(y, classifiers)
    plt.gca().invert_yaxis()  # Best model on top
    plt.tight_layout()
    plt.savefig("Result/Computational_Resources_GFLOPs_barh.png", dpi=1700)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.barh(y, inference_speed, color=colors2)
    plt.ylabel("Models", fontsize=13)
    plt.xlabel("Inference Speed (ms/sample)", fontsize=13)
    plt.title("Inference Speed Comparison", fontsize=14)
    plt.yticks(y, classifiers)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("Result/Inference_Speed_ms_barh.png", dpi=1700)
    plt.show()
