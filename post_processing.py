"""
    Read all confusion matrices, and
    calculate the average performance.

"""

import os
import pandas as pd
from train import algo
from load_data import load_train_test


def parse_confusion_matrix(file_path):
    data = {}
    with open(file_path, "r") as file:
        for line in file:
            if line.strip():
                key, value = line.strip().split(": ")
                data[key] = float(value)
    return data


def calculate_average(file_paths):
    total_counts = {}
    num_files = len(file_paths)

    # Initialize total_counts with zeros
    with open(file_paths[0], "r") as file:
        for line in file:
            if line.strip():
                key, _ = line.strip().split(": ")
                total_counts[key] = 0.0

    # Sum counts from all files
    for file_path in file_paths:
        data = parse_confusion_matrix(file_path)
        for key, value in data.items():
            total_counts[key] += value

    # Calculate average
    for key in total_counts:
        total_counts[key] /= num_files

    return total_counts


def save_average_to_file(average_counts, output_file):
    with open(output_file, "w") as file:
        for key, value in average_counts.items():
            file.write(f"{key}: {value}\n")


def generate_average_result(model_name):
    # Directory containing the text files
    folder_path = f"output/model/{model_name}/"

    file_paths = [
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if filename.endswith(".txt")
    ]

    average_counts = calculate_average(file_paths)

    # Save average counts to a new file
    output_file = os.path.join(folder_path, "confusion_matrix_avg.txt")
    save_average_to_file(average_counts, output_file)


def get_accuracy_values(folder_path):
    accuracy_values = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == "confusion_matrix_avg.txt":
                file_path = os.path.join(root, file)
                data = parse_confusion_matrix(file_path)
                accuracy_values[root] = data
    return accuracy_values


def calculate_overall_accuracy(true_pos, true_neg, total):
    return (true_pos + true_neg) / total


def calculate_overall_accuracies(data):
    data["Test Snippets - Overall Accuracy"] = calculate_overall_accuracy(
        data["Test Snippets - True Pos"],
        data["Test Snippets - True Neg"],
        data["Test Snippets - True Pos"]
        + data["Test Snippets - True Neg"]
        + data["Test Snippets - False Pos"]
        + data["Test Snippets - False Neg"],
    )
    data["Test Songs - Overall Accuracy"] = calculate_overall_accuracy(
        data["Test Songs - True Pos"],
        data["Test Songs - True Neg"],
        data["Test Songs - True Pos"]
        + data["Test Songs - True Neg"]
        + data["Test Songs - False Pos"]
        + data["Test Songs - False Neg"],
    )
    return data


def compare_results():
    folder_path = "output/model/"
    accuracy_values = get_accuracy_values(folder_path)

    processed_accuracy_values = {
        folder: calculate_overall_accuracies(data)
        for folder, data in accuracy_values.items()
    }

    # Convert to DataFrame
    df = pd.DataFrame(processed_accuracy_values).transpose()
    df.to_csv("output/model/results_comparison.csv")


## MAIN FUNCTION
def run(model_dict, total_run, use_long=False):
    (
        train_dataloader,
        test_dataloader,
        train_size,
        batch_size_training,
        batch_size_testing,
    ) = load_train_test(use_long)

    for model_name, model in model_dict.items():
        for i in range(total_run):
            algo(
                model,
                model_name,
                train_dataloader,
                test_dataloader,
                train_size,
                batch_size_training,
                batch_size_testing,
                use_long,
                run_num=i,
            )
        generate_average_result(model_name)
