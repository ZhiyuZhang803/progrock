"""
    Load both train and test data.

"""

import json
import logging
import numpy as np
from Dataset import MyDataset
from torch.utils.data import DataLoader

from load_data_only_train import load_data_only_train

logger = logging.getLogger(__name__)


def read_json(input_path):
    # Save Sample
    data_temp = []
    label_temp = []
    name_temp = []
    snippets_temp = []

    # Read in the data
    with open(input_path, "r") as file:
        data = json.load(file)

    # Save features
    for i in range(len(data["Name"])):
        data_temp.append(
            np.array(
                data["MFCC"][i]
                + data["Spectrogram"][i]
                + data["Chromagram"][i]
                + [data["Onset"][i]]
            )
        )
        label_temp.append(int(data["Label"][i] == 0))
        name_temp.append(data["Name"][i])
        snippets_temp.append(data["Snippets"][i])

    # Release the memory
    del data

    return (
        data_temp,
        label_temp,
        name_temp,
        snippets_temp,
    )


def load_data(
    non_prog_other_path_train,
    non_prog_pop_path_train,
    prog_path_train,
    non_prog_other_path_test,
    non_prog_pop_path_test,
    prog_path_test,
):
    train_data = []
    train_label = []
    train_name = []
    train_snippets = []
    test_data = []
    test_label = []
    test_name = []
    test_snippets = []

    for i in [prog_path_train, non_prog_other_path_train, non_prog_pop_path_train]:
        (
            train_data_temp,
            train_label_temp,
            train_name_temp,
            train_snippets_temp,
        ) = read_json(i)
        train_data += train_data_temp
        train_label += train_label_temp
        train_name += train_name_temp
        train_snippets += train_snippets_temp
        logger.debug(f"Loaded train: {i}")

    for i in [prog_path_test, non_prog_other_path_test, non_prog_pop_path_test]:
        (
            test_data_temp,
            test_label_temp,
            test_name_temp,
            test_snippets_temp,
        ) = read_json(i)
        test_data += test_data_temp
        test_label += test_label_temp
        test_name += test_name_temp
        test_snippets += test_snippets_temp
        logger.debug(f"Loaded test: {i}")

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


def load_train_test(only_train=False, fix_random=False):
    # Set the random seed for PyTorch
    if fix_random:
        np.random.seed(1234)

    non_prog_other_path_train = "../data/Feature_Extraction_Other.json"
    non_prog_pop_path_train = "../data/Feature_Extraction_Top_Pop.json"
    prog_path_train = "../data/Feature_Extraction_Prog.json"

    if only_train:
        (
            train_data,
            train_label,
            test_data,
            test_label,
            train_name_float,
            test_name_float,
            train_snippets,
            test_snippets,
        ) = load_data_only_train(
            non_prog_other_path_train, non_prog_pop_path_train, prog_path_train
        )
    else:
        non_prog_other_path_test = "../data/Test_Feature_Extraction_Other.json"
        non_prog_pop_path_test = "../data/Test_Feature_Extraction_Top_Pop.json"
        prog_path_test = "../data/Test_Feature_Extraction_Prog.json"

        (
            train_data,
            train_label,
            test_data,
            test_label,
            train_name_float,
            test_name_float,
            train_snippets,
            test_snippets,
        ) = load_data(
            non_prog_other_path_train,
            non_prog_pop_path_train,
            prog_path_train,
            non_prog_other_path_test,
            non_prog_pop_path_test,
            prog_path_test,
        )

    batch_size_training = 500
    batch_size_testing = len(test_data)
    train_size = len(train_data)

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

    return (
        train_dataloader,
        test_dataloader,
        train_size,
        batch_size_training,
        batch_size_testing,
    )
