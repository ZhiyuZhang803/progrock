from SongList import SongList
import logging

logging.basicConfig(filename='log_file.log',
                    encoding='utf-8',
                    format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)

logger.debug("===============  Started!  ===============")

directory = "../CAP6610SP24_training_set"
# directory = "../toy_training_set"

train_list = SongList(directory)
