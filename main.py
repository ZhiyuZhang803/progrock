import logging

logging.basicConfig(
    filename="output/log_file.log",
    encoding="utf-8",
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)

logger.debug("===============  Started!  ===============")


from train import algo
from Models.Baseline import Baseline
from Models.ModifiedBaseline import ModifiedBaseline
from Models.DeepBaseline import DeepBaseline
from Models.DepthWiseBaseline import DepthWiseBaseline
from Models.AcousticModel import AcousticModel
from Models.GenreModel import GenreModel


# algo(Baseline(), "baseline")
# algo(ModifiedBaseline(), "modifiedbaseline")
# algo(DeepBaseline(), "deepbaseline")
# algo(DepthWiseBaseline(), "depthwisebaseline")
# algo(AcousticModel(), "acoustic")
algo(GenreModel(), "genre")
