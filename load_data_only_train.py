"""
    Load only train data, and 
    random select 20% as test set.

"""

import json
import logging
import numpy as np
import random

logger = logging.getLogger(__name__)


def read_json(input_path, fix_random):
    if fix_random:
        random.seed(1234)
        np.random.seed(1234)

    # Save Sample
    train_data_temp = []
    train_label_temp = []
    test_data_temp = []
    test_label_temp = []
    train_name_temp = []
    test_name_temp = []
    train_snippets_temp = []
    test_snippets_temp = []

    # Read in the data
    with open(input_path, "r") as file:
        data = json.load(file)

    # Randomly Selet 80% songs as training, 20% songs as testing
    number_sample = len(list(set(data["Name"])))
    select_name = list(set(data["Name"]))
    random.shuffle(select_name)
    training_name = select_name[: int(0.8 * number_sample)]
    testing_name = select_name[int(0.8 * number_sample) :]

    # Save the train and test features
    for i in range(len(data["Name"])):
        if data["Name"][i] in training_name:
            train_data_temp.append(
                np.array(
                    data["MFCC"][i]
                    + data["Spectrogram"][i]
                    + data["Chromagram"][i]
                    + [data["Onset"][i]]
                )
            )
            train_label_temp.append(int(data["Label"][i] == 0))
            train_name_temp.append(data["Name"][i])
            train_snippets_temp.append(data["Snippets"][i])
        elif data["Name"][i] in testing_name:
            test_data_temp.append(
                np.array(
                    data["MFCC"][i]
                    + data["Spectrogram"][i]
                    + data["Chromagram"][i]
                    + [data["Onset"][i]]
                )
            )
            test_label_temp.append(int(data["Label"][i] == 0))
            test_name_temp.append(data["Name"][i])
            test_snippets_temp.append(data["Snippets"][i])
    # Release the memory
    del data

    return (
        train_data_temp,
        train_label_temp,
        train_name_temp,
        train_snippets_temp,
        test_data_temp,
        test_label_temp,
        test_name_temp,
        test_snippets_temp,
    )


def load_data_only_train(non_prog_other_path, non_prog_pop_path, prog_path, fix_random):
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    train_name = []
    test_name = []
    train_snippets = []
    test_snippets = []

    for i in [prog_path, non_prog_other_path, non_prog_pop_path]:
        (
            train_data_temp,
            train_label_temp,
            train_name_temp,
            train_snippets_temp,
            test_data_temp,
            test_label_temp,
            test_name_temp,
            test_snippets_temp,
        ) = read_json(i, fix_random)
        train_data += train_data_temp
        train_label += train_label_temp
        test_data += test_data_temp
        test_label += test_label_temp
        train_name += train_name_temp
        test_name += test_name_temp
        train_snippets += train_snippets_temp
        test_snippets += test_snippets_temp
        logger.debug(f"Loaded: {i}")

    logger.debug(f"train size = {len(train_data)}, test size = {len(test_data)}")
    logger.debug(
        f"train song = {len(set(train_name))}, test song = {len(set(test_name))}"
    )

    # Transform name list from string to float
    # Easy to handle and use in PyTorch
    train_name_dict = {}
    test_name_dict = {}
    train_name_float = []
    test_name_float = []

    # Set count flags
    flag = 0
    for i in train_name:
        if i not in list(train_name_dict.values()):
            flag += 1
            train_name_dict[flag] = i
            train_name_float.append(flag)
        else:
            train_name_float.append(flag)

    for i in test_name:
        if i not in list(test_name_dict.values()):
            flag += 1
            test_name_dict[flag] = i
            test_name_float.append(flag)
        else:
            test_name_float.append(flag)

    return (
        train_data,
        train_label,
        test_data,
        test_label,
        train_name_float,
        test_name_float,
        train_snippets,
        test_snippets,
    )
