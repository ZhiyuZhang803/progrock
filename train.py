import os
import time
import torch
import pickle
import logging
import pandas as pd
import torch.nn as nn

logger = logging.getLogger(__name__)


def algo(
    model,
    model_name,
    train_dataloader,
    test_dataloader,
    train_size,
    batch_size_training,
    batch_size_testing,
    run_num=0,
    fix_random=False,
):
    tik = time.time()

    # Set the random seed for PyTorch
    if fix_random:
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        torch.mps.manual_seed(1234)

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.debug(f"Device: {device}")

    # model = Baseline()
    model.to(device)

    logger.debug(model)

    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Simple training process
    train_epoch = int(train_size / batch_size_training) + 1
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

    # Create the directory if it doesn't exist
    output_dir = f"output/model/{model_name}/"
    os.makedirs(output_dir, exist_ok=True)

    # Save model in a pickle file
    model_file = output_dir + f"model_{run_num}.pkl"
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
    df.to_csv(output_dir + f"test_result_{run_num}.csv")

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
        logger.info(f"Non-Prog Accuracy: {pred_accuracy[0] / count_class[0]}")
        logger.info(f"Prog Accuracy: {pred_accuracy[1] / count_class[1]}")

    with open(output_dir + f"confusion_matrix_{run_num}.txt", "w") as file:
        file.write("Train Snippets - True Neg: {}\n".format(pred_accuracy[0]))
        file.write(
            "Train Snippets - False Neg: {}\n".format(count_class[1] - pred_accuracy[1])
        )
        file.write("Train Snippets - True Pos: {}\n".format(pred_accuracy[1]))
        file.write(
            "Train Snippets - False Pos: {}\n".format(count_class[0] - pred_accuracy[0])
        )
        file.write(
            f"Train Snippets - Non-Prog Accuracy: {pred_accuracy[0] / count_class[0]}\n"
        )
        file.write(
            f"Train Snippets - Prog Accuracy: {pred_accuracy[1] / count_class[1]}\n"
        )

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

    with open(output_dir + f"confusion_matrix_{run_num}.txt", "a") as file:
        file.write("\n")
        file.write("Train Songs - True Neg: {}\n".format(true_neg))
        file.write("Train Songs - False Neg: {}\n".format(false_neg))
        file.write("Train Songs - True Pos: {}\n".format(true_pos))
        file.write("Train Songs - False Pos: {}\n".format(false_pos))
        file.write(
            f"Train Songs - Non-Prog Accuracy: {true_neg/(true_neg + false_pos)}\n"
        )
        file.write(f"Train Songs - Prog Accuracy: {true_pos/(true_pos + false_neg)}\n")

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

        logger.info(f"Non-Prog Accuracy: {pred_accuracy[0] / count_class[0]}")
        logger.info(f"Prog Accuracy: {pred_accuracy[1] / count_class[1]}")

    with open(output_dir + f"confusion_matrix_{run_num}.txt", "a") as file:
        file.write("\n")
        file.write("Test Snippets - True Neg: {}\n".format(pred_accuracy[0]))
        file.write(
            "Test Snippets - False Neg: {}\n".format(count_class[1] - pred_accuracy[1])
        )
        file.write("Test Snippets - True Pos: {}\n".format(pred_accuracy[1]))
        file.write(
            "Test Snippets - False Pos: {}\n".format(count_class[0] - pred_accuracy[0])
        )
        file.write(
            f"Test Snippets - Non-Prog Accuracy: {pred_accuracy[0] / count_class[0]}\n"
        )
        file.write(
            f"Test Snippets - Prog Accuracy: {pred_accuracy[1] / count_class[1]}\n"
        )

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

    with open(output_dir + f"confusion_matrix_{run_num}.txt", "a") as file:
        file.write("\n")
        file.write("Test Songs - True Neg: {}\n".format(true_neg))
        file.write("Test Songs - False Neg: {}\n".format(false_neg))
        file.write("Test Songs - True Pos: {}\n".format(true_pos))
        file.write("Test Songs - False Pos: {}\n".format(false_pos))
        file.write(
            f"Test Songs - Non-Prog Accuracy: {true_neg/(true_neg + false_pos)}\n"
        )
        file.write(f"Test Songs - Prog Accuracy: {true_pos/(true_pos + false_neg)}")

    tok = time.time()
    logger.info(f"Elapsed time: {tok - tik} seconds")
