import torch
import torch.nn as nn
import numpy as np
import logging

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

# Prepare Training
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
train_dataloader = DataLoader(my_dataset, batch_size=batch_size_training, shuffle=True)
test_dataloader = DataLoader(my_dataset2, batch_size=batch_size_testing, shuffle=True)

model = Baseline()
model.to(device)

logger.debug(model)

error = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
