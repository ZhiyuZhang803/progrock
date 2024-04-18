import time
import torch
import pickle
import logging
import numpy as np
import pandas as pd
import torch.nn as nn

from Dataset import MyDataset
from torch.utils.data import DataLoader
from Models.Baseline import Baseline
from load_data import load_data

logging.basicConfig(
    filename="output/log_file.log",
    encoding="utf-8",
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)

logger.debug("===============  Started!  ===============")


def pretrain(model, model_name):
    tik = time.time()
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.debug(f"Device: {device}")

    non_prog_other_path = "../data/Feature_Extraction_Other.json"
    non_prog_pop_path = "../data/Feature_Extraction_Top_Pop.json"
    prog_path = "../data/Feature_Extraction_Prog.json"

    (
        train_data,
        train_label,
        test_data,
        test_label,
        train_name_float,
        test_name_float,
        train_snippets,
        test_snippets,
    ) = load_data(non_prog_other_path, non_prog_pop_path, prog_path)

    batch_size_training = 500
    batch_size_testing = len(test_data)

    my_dataset = MyDataset(
        np.array(train_data),
        np.array(train_label),
        np.array(train_name_float),
        np.array(train_snippets),
    )
    my_dataset2 = MyDataset(
        np.array(test_data),
        np.array(test_label),
        np.array(test_name_float),
        np.array(test_snippets),
    )

    # Create DataLoader
    train_dataloader = DataLoader(
        my_dataset, batch_size=batch_size_training, shuffle=True
    )
    test_dataloader = DataLoader(
        my_dataset2, batch_size=batch_size_testing, shuffle=True
    )

    # model = Baseline()
    model.to(device)

    logger.debug(model)

    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Simple training process
    train_epoch = int(len(train_data) / batch_size_training) + 1
    for epoch in range(train_epoch):
        for i, (images, labels, names, snippets) in enumerate(train_dataloader):
            # Transfering images and labels to GPU if available
            images, labels = images.to(device), labels.to(device)

            train = images
            # Forward pass
            outputs = model(train)
            loss = error(outputs, labels)

            # Initializing a gradient as 0 so there is no mixing of gradient among the batches
            optimizer.zero_grad()

            # Propagating the error backward
            loss.backward()

            # Optimizing the parameters
            optimizer.step()

        logger.info(f"Epoch: {epoch}, Loss: {loss.data:.5f}")

    # Save model in a pickle file
    model_file = f"output/model/{model_name}.pkl"
    with open(model_file, "wb") as file:
        pickle.dump(model, file)
        logger.debug(f"Model saved to {model_file}")

    # Simple test process
    pred_accuracy = {0: 0, 1: 0}
    count_class = {0: 0, 1: 0}
    true_class = []
    predicted_class = []
    predicted_names = []
    predicted_snippets = []

    with torch.no_grad():
        for images, labels, names, snippets in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            test = images
            outputs = model(test)

            # Record the accuracy for prog and non-prog seperately
            for i in range(batch_size_testing):
                l = outputs[i].tolist()
                if l[0] >= l[1]:
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
        logger.info(f"Non-Prog Accuracy: {pred_accuracy[0] / count_class[0]}")
        logger.info(f"Prog Accuracy: {pred_accuracy[1] / count_class[1]}")

    # Save test result
    df = pd.DataFrame(
        {
            "name": predicted_names,
            "snippet": predicted_snippets,
            "predicted_class": predicted_class,
            "true_class": true_class,
        }
    )
    df.to_csv(f"output/model/{model_name}_test_result.csv")

    # See how many songs are correctly predicted
    song_predict = []
    song_true = []

    for i in set(df["name"]):
        df_temp = df[df["name"] == i]
        predict_list = list(df_temp["predicted_class"])
        true_list = list(df_temp["true_class"])
        if predict_list.count(1) >= predict_list.count(0):
            song_predict.append(1)
        else:
            song_predict.append(0)
        if sum(true_list) == 0:
            song_true.append(0)
        else:
            song_true.append(1)

    song_accuracy = sum(
        [1 if song_predict[i] == song_true[i] else 0 for i in range(len(song_true))]
    ) / len(song_true)
    logger.info(f"Song Accuracy: {song_accuracy}")

    # On Training Snippets
    logger.info("On training snippets")
    pred_accuracy = {0: 0, 1: 0}
    count_class = {0: 0, 1: 0}
    true_class = []
    predicted_class = []
    predicted_names = []
    predicted_snippets = []

    with torch.no_grad():
        for images, labels, names, snippets in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            test = images
            outputs = model(test)

            # Record the accuracy for prog and non-prog seperately
            for i in range(batch_size_training):
                try:
                    l = outputs[i].tolist()
                    if l[0] >= l[1]:
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
        print("Non-Prog Accuracy: ", pred_accuracy[0] / count_class[0])
        print("Prog Accuracy: ", pred_accuracy[1] / count_class[1])

    with open(f"output/model/{model_name}_confusion_matrix.txt", "w") as file:
        file.write("Train Snippets:\n")
        file.write("True Neg: {}\n".format(pred_accuracy[0]))
        file.write("False Neg: {}\n".format(count_class[1] - pred_accuracy[1]))
        file.write("True Pos: {}\n".format(pred_accuracy[1]))
        file.write("False Pos: {}\n".format(count_class[0] - pred_accuracy[0]))

    # On Training Songs
    logger.info("On training songs")
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
        if predict_list.count(1) >= predict_list.count(0):
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

    logger.info(f"Non-Prog Accuracy: {true_neg/(true_neg + false_pos)}")
    logger.info(f"Prog Accuracy: {true_pos/(true_pos + false_neg)}")

    with open(f"output/model/{model_name}_confusion_matrix.txt", "a") as file:
        file.write("\nTrain Songs:\n")
        file.write("True Neg: {}\n".format(true_neg))
        file.write("False Neg: {}\n".format(false_neg))
        file.write("True Pos: {}\n".format(true_pos))
        file.write("False Pos: {}\n".format(false_pos))

    # On Testing (Validation) Snippets
    logger.info("On test snippets")
    pred_accuracy = {0: 0, 1: 0}
    count_class = {0: 0, 1: 0}
    true_class = []
    predicted_class = []
    predicted_names = []
    predicted_snippets = []

    with torch.no_grad():
        for images, labels, names, snippets in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            test = images
            outputs = model(test)

            # Record the accuracy for prog and non-prog seperately
            for i in range(batch_size_testing):
                l = outputs[i].tolist()
                if l[0] >= l[1]:
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

        print("Non-Prog Accuracy: ", pred_accuracy[0] / count_class[0])
        print("Prog Accuracy: ", pred_accuracy[1] / count_class[1])

    with open(f"output/model/{model_name}_confusion_matrix.txt", "a") as file:
        file.write("\nTest Snippets:\n")
        file.write("True Neg: {}\n".format(pred_accuracy[0]))
        file.write("False Neg: {}\n".format(count_class[1] - pred_accuracy[1]))
        file.write("True Pos: {}\n".format(pred_accuracy[1]))
        file.write("False Pos: {}\n".format(count_class[0] - pred_accuracy[0]))

    # On Testing Songs
    logger.info("On testing songs")
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
        if predict_list.count(1) >= predict_list.count(0):
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

    logger.info(f"Non-Prog Accuracy: {true_neg/(true_neg + false_pos)}")
    logger.info(f"Prog Accuracy: {true_pos/(true_pos + false_neg)}")

    with open(f"output/model/{model_name}_confusion_matrix.txt", "a") as file:
        file.write("\nTest Songs:\n")
        file.write("True Neg: {}\n".format(true_neg))
        file.write("False Neg: {}\n".format(false_neg))
        file.write("True Pos: {}\n".format(true_pos))
        file.write("False Pos: {}\n".format(false_pos))

    tok = time.time()
    logger.info(f"Elapsed time: {tok - tik} seconds")


pretrain(Baseline(), "baseline")
