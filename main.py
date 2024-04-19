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


# algo(Baseline(), "baseline")
algo(Baseline(), "baseline")
