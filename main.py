import logging

logging.basicConfig(
    filename="output/log_file.log",
    encoding="utf-8",
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)

logger.debug("===============  Started!  ===============")

from post_processing import run, generate_average_result, compare_results
from Models.Baseline import Baseline
from Models.ModifiedBaseline import ModifiedBaseline
from Models.DeepBaseline import DeepBaseline
from Models.DepthWiseBaseline import DepthWiseBaseline
from Models.AcousticModel import AcousticModel
from Models.GenreModel import GenreModel
from Models.ResNet1D import ResNet1D


model_dict = {
    # "baseline": Baseline(),
    # "modifiedbaseline": ModifiedBaseline(),
    # "deepbaseline": DeepBaseline(),
    # "depthwisebaseline": DepthWiseBaseline(),
    # "acoustic": AcousticModel(),
    "genre": GenreModel(),
    "resnet": ResNet1D(),
}


total_run = 10

## MAIN FUNCTION
# run(model_dict, total_run, use_long=True)

## ADD LONG AVERAGE
# model_long_names = [
#     "baseline_long",
#     "modifiedbaseline_long",
#     "deepbaseline_long",
#     "depthwisebaseline_long",
#     "acoustic_long",
#     "genre_long",
#     "resnet_long",
# ]

# for name in model_long_names:
#     generate_average_result(name)

## COMPARE RESULTS
compare_results()
