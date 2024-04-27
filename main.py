import logging

logging.basicConfig(
    filename="output/log_file.log",
    encoding="utf-8",
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)

logger.debug("===============  Started!  ===============")

from post_processing import run
from Models.Baseline import Baseline
from Models.ModifiedBaseline import ModifiedBaseline
from Models.DeepBaseline import DeepBaseline
from Models.DepthWiseBaseline import DepthWiseBaseline
from Models.AcousticModel import AcousticModel
from Models.GenreModel import GenreModel
from Models.ResNet1D import ResNet1D


model_dict = {
    "baseline": Baseline(),
    "modifiedbaseline": ModifiedBaseline(),
    "deepbaseline": DeepBaseline(),
    "depthwisebaseline": DepthWiseBaseline(),
    "acoustic": AcousticModel(),
    "genre": GenreModel(),
    "resnet": ResNet1D(),
}

total_run = 10

run(model_dict, total_run)
