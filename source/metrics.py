import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    multilabel_confusion_matrix,
)

from source import config
from source.data import explode_locc
from source.utils import hash_dict


def calculate_flat_binary_metrics(
    y_true,
    y_pred,
    labels,
    model_type: str,
    model_name: str,
    hyperparams: dict = {},
    save: bool = False,
    evals_result: dict = {},
) -> str:
    model_name = f"{model_name}_{hash_dict(hyperparams)}"
    metrics_path = config.METRICS_DIR / model_type / model_name
    metrics_path.mkdir(parents=True, exist_ok=True)

    if hyperparams:
        with (metrics_path / "hyperparams.json").open("w") as f:
            json.dump(hyperparams, f, indent=4)

    if evals_result:
        with (metrics_path / "evals_result.json").open("w") as f:
            json.dump(evals_result, f, indent=4)

    classification_report_dict = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        zero_division=0,
        output_dict=True,
    )
    report = {
        "accuracy": accuracy_score(y_true, y_pred),
        **classification_report_dict,
    }

    if save:
        with metrics_path.joinpath("report.json").open("w") as f:
            json.dump(report, f, indent=4)

    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

    print("\nMultilabel Confusion Matrices (one for each label):")
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    for i, cls in enumerate(labels):
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            mcm[i],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted Negative", "Predicted Positive"],
            yticklabels=["Actual Negative", "Actual Positive"],
        )
        plt.title(f"Confusion Matrix for label: {cls}")
        plt.ylabel("Actual Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        if save:
            plt.savefig(metrics_path / f"confusion_matrix_{cls}.png")
        plt.show()

    return (
        classification_report_dict["weighted avg"]["f1-score"],
        model_name,
        metrics_path,
    )


def draw_metrics_histograms():
    models = [
        config.METRICS_DIR / "nbayes" / "vectorizer_4b8e43f1bd58cfe525347ab804a3b4b4",
        config.METRICS_DIR
        / "xgboost"
        / "multilingual_99914b932bd37a50b983c5e7c90ae93b",
        config.METRICS_DIR / "xgbhi" / "multi_002d3d0c283f91ff93a479eb3146a7f2",
    ]

    model_names = [
        "Naive Bayes",
        "XGBoost + multilingual-MiniLM-L12-v2",
        "Hierarchical XGBoost + multilingual-MiniLM-L12-v2",
    ]

    metrics_to_plot = ["f1-score", "precision", "recall"]
    class_labels = ["AP", "B", "D", "DI", "O", "P", "PR", "PS", "PZ", "Q"]

    loaded_reports = []
    for model_path in models:
        try:
            with (model_path / "report.json").open("r") as f:
                report = json.load(f)
            loaded_reports.append(report)
        except FileNotFoundError:
            print(f"Warning: Report not found for model path {model_path}")
            loaded_reports.append({})

    plot_data_list = []
    for i, report_dict in enumerate(loaded_reports):
        if not report_dict:
            continue
        model_name_for_plot = model_names[i]
        for label_name in class_labels:
            if label_name in report_dict:
                for metric_name in metrics_to_plot:
                    if (
                        isinstance(report_dict[label_name], dict)
                        and metric_name in report_dict[label_name]
                    ):
                        plot_data_list.append(
                            {
                                "model": model_name_for_plot,
                                "class": label_name,
                                "metric_type": metric_name,
                                "value": report_dict[label_name][metric_name],
                            }
                        )

    if not plot_data_list:
        print("No data available for plotting. Please check report contents and paths.")
        return

    plot_df = pd.DataFrame(plot_data_list)

    for metric_name_to_display in metrics_to_plot:
        plt.figure(figsize=(18, 8))

        metric_specific_df = plot_df[plot_df["metric_type"] == metric_name_to_display]

        if metric_specific_df.empty:
            print(f"No data for metric '{metric_name_to_display}' to plot.")
            continue

        sns.barplot(
            x="class",
            y="value",
            hue="model",
            data=metric_specific_df,
            palette="viridis",
        )

        plt.title(
            f"Class-wise {metric_name_to_display.capitalize()} Comparison Across Models",
            fontsize=16,
        )
        plt.xlabel("Class", fontsize=14)
        plt.ylabel(metric_name_to_display.capitalize(), fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(
            title="Model",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=10,
            title_fontsize=12,
        )
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        save_path = (
            config.METRICS_DIR
            / f"classwise_{metric_name_to_display.replace('-','_')}_comparison.png"
        )
        try:
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
        except Exception as e:
            print(f"Error saving plot {save_path}: {e}")
        plt.show()


if __name__ == "__main__":
    draw_metrics_histograms()
