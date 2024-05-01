import math
import torch
import pickle
import logging
import pandas as pd
from load_data import load_train_test

logger = logging.getLogger(__name__)


def analyze_results(
    model_name,
    current_model,
    data_loader,
    sigmoid_cutoff,
    final_cutoff,
    fileoutput,
    device,
    batch_size_training,
):
    # On Snippets
    pred_accuracy = {0: 0, 1: 0}
    count_class = {0: 0, 1: 0}
    true_class = []
    predicted_class = []
    predicted_names = []
    predicted_snippets = []
    # print("Sigmoid Cutoff:", sigmoid_cutoff, "Final Cutoff:", final_cutoff)
    with torch.no_grad():
        for images, labels, names, snippets in data_loader:
            images, labels = images.to(device), labels.to(device)
            test = images
            outputs = current_model(test)

            # Record the accuracy for prog and non-prog seperately
            for i in range(batch_size_training):
                try:
                    l = outputs[i].tolist()
                    if math.exp(l[0]) / (math.exp(l[0]) + math.exp(l[1])) >= (
                        1 - sigmoid_cutoff
                    ):
                        predict_out = 0
                    else:
                        predict_out = 1
                    true_class.append(labels[i].item())
                    predicted_class.append(predict_out)
                    predicted_names.append(names[i].item())
                    predicted_snippets.append(snippets[i].item())
                    true_out = labels[i].item()
                    count_class[true_out] += 1
                    if true_out == predict_out:
                        pred_accuracy[true_out] += 1
                except:
                    break

    df = pd.DataFrame(
        {
            "name": predicted_names,
            "snippet": predicted_snippets,
            "predicted_class": predicted_class,
            "true_class": true_class,
        }
    )

    song_predict = []
    song_true = []

    for i in set(df["name"]):
        df_temp = df[df["name"] == i]
        predict_list = list(df_temp["predicted_class"])
        true_list = list(df_temp["true_class"])
        if (
            predict_list.count(1) / (predict_list.count(0) + predict_list.count(1))
        ) >= final_cutoff:
            song_predict.append(1)
        else:
            song_predict.append(0)
        if sum(true_list) == 0:
            song_true.append(0)
        else:
            song_true.append(1)

    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    for i in range(len(song_true)):
        if (song_true[i] == 1) and (song_predict[i] == 1):
            true_pos += 1
        elif (song_true[i] == 1) and (song_predict[i] == 0):
            false_neg += 1
        elif (song_true[i] == 0) and (song_predict[i] == 1):
            false_pos += 1
        else:
            true_neg += 1
    info = [
        model_name,
        str(sigmoid_cutoff),
        str(final_cutoff),
        str(round(pred_accuracy[0] * 100 / count_class[0], 2)),
        str(round(pred_accuracy[1] * 100 / count_class[1], 2)),
        str(round(pred_accuracy[0], 2)),
        str(round(count_class[1] - pred_accuracy[1], 2)),
        str(round(pred_accuracy[1], 2)),
        str(round(count_class[0] - pred_accuracy[0], 2)),
        str(round(true_neg * 100 / (true_neg + false_pos), 2)),
        str(round(true_pos * 100 / (true_pos + false_neg))),
        str(true_neg),
        str(false_neg),
        str(true_pos),
        str(false_pos),
        str(
            round(
                (true_neg + true_pos)
                * 100
                / (true_neg + true_pos + false_neg + false_pos),
                2,
            )
        ),
    ]
    info = ",".join(info) + "\n"
    fileoutput.write(info)


import math


def analyze_results_2(
    model_name,
    current_model,
    data_loader,
    sigmoid_cutoff,
    final_cutoff,
    fileoutput,
    label_file,
    device,
    batch_size_testing,
):
    # On Snippets
    pred_accuracy = {0: 0, 1: 0}
    count_class = {0: 0, 1: 0}
    true_class = []
    predicted_class = []
    predicted_names = []
    predicted_snippets = []
    # print("Sigmoid Cutoff:", sigmoid_cutoff, "Final Cutoff:", final_cutoff)
    with torch.no_grad():
        for images, labels, names, snippets in data_loader:
            images, labels = images.to(device), labels.to(device)
            test = images
            outputs = current_model(test)

            # Record the accuracy for prog and non-prog seperately
            for i in range(batch_size_testing):
                l = outputs[i].tolist()
                if math.exp(l[0]) / (math.exp(l[0]) + math.exp(l[1])) >= (
                    1 - sigmoid_cutoff
                ):
                    predict_out = 0
                else:
                    predict_out = 1
                true_class.append(labels[i].item())
                predicted_class.append(predict_out)
                predicted_names.append(names[i].item())
                predicted_snippets.append(snippets[i].item())
                true_out = labels[i].item()
                count_class[true_out] += 1
                if true_out == predict_out:
                    pred_accuracy[true_out] += 1

    df = pd.DataFrame(
        {
            "name": predicted_names,
            "snippet": predicted_snippets,
            "predicted_class": predicted_class,
            "true_class": true_class,
        }
    )

    song_predict = []
    song_true = []

    name = []
    pred_perc = []
    for i in set(df["name"]):
        df_temp = df[df["name"] == i]
        predict_list = list(df_temp["predicted_class"])
        true_list = list(df_temp["true_class"])
        if (
            predict_list.count(1) / (predict_list.count(0) + predict_list.count(1))
        ) >= final_cutoff:
            song_predict.append(1)
        else:
            song_predict.append(0)
        if sum(true_list) == 0:
            song_true.append(0)
        else:
            song_true.append(1)
        name.append(i)
        pred_perc.append(
            round(
                predict_list.count(1) / (predict_list.count(0) + predict_list.count(1)),
                2,
            )
        )

    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    for i in range(len(song_true)):
        if (song_true[i] == 1) and (song_predict[i] == 1):
            true_pos += 1
        elif (song_true[i] == 1) and (song_predict[i] == 0):
            false_neg += 1
        elif (song_true[i] == 0) and (song_predict[i] == 1):
            false_pos += 1
        else:
            true_neg += 1

    # Write the results
    info = [
        model_name,
        str(sigmoid_cutoff),
        str(final_cutoff),
        str(round(pred_accuracy[0] * 100 / count_class[0], 2)),
        str(round(pred_accuracy[1] * 100 / count_class[1], 2)),
        str(round(pred_accuracy[0], 2)),
        str(round(count_class[1] - pred_accuracy[1], 2)),
        str(round(pred_accuracy[1], 2)),
        str(round(count_class[0] - pred_accuracy[0], 2)),
        str(round(true_neg * 100 / (true_neg + false_pos), 2)),
        str(round(true_pos * 100 / (true_pos + false_neg))),
        str(true_neg),
        str(false_neg),
        str(true_pos),
        str(false_pos),
        str(
            round(
                (true_neg + true_pos)
                * 100
                / (true_neg + true_pos + false_neg + false_pos),
                2,
            )
        ),
    ]
    info = ",".join(info) + "\n"
    fileoutput.write(info)
    # Save the label
    df_final = pd.DataFrame(
        {
            "name": name,
            "true_label": song_true,
            "pred_label": song_predict,
            "prec": pred_perc,
        }
    )
    df_final.to_csv(label_file, index=False)


def cutoff_eval(model_to_eval="baseline"):
    # check device
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.debug(f"Device: {device}")

    # load data
    (
        train_dataloader,
        test_dataloader,
        train_size,
        batch_size_training,
        batch_size_testing,
    ) = load_train_test(use_long=False)

    # folder that saves the pre-trained models
    save_path = f"output/model/{model_to_eval}/"

    # Please prepare train_dataloader and test_dataloader before running the model
    with open(save_path + "output.txt", "w") as file_output:
        for i in range(10):
            model_name = "model_" + str(i) + ".pkl"

            with open(save_path + model_name, "rb") as file:
                model = pickle.load(file)
            for cut_sigmoid in [0.5, 0.6, 0.7]:
                for cut_final in [0.5, 0.55, 0.6]:
                    # BEFORE RUN THE CODE, PLEASE CREATE A FOLDER NAME "final_label"
                    label_file = (
                        save_path
                        + "final_label/"
                        + "model_"
                        + str(i)
                        + "_"
                        + str(cut_sigmoid)
                        + "_"
                        + str(cut_final)
                        + "_output.csv"
                    )
                    analyze_results(
                        model_name,
                        model,
                        train_dataloader,
                        cut_sigmoid,
                        cut_final,
                        file_output,
                        device,
                        batch_size_training,
                    )
                    analyze_results_2(
                        model_name,
                        model,
                        test_dataloader,
                        cut_sigmoid,
                        cut_final,
                        file_output,
                        label_file,
                        device,
                        batch_size_testing,
                    )
