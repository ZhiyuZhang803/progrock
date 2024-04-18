from SongList import SongList
import logging

logging.basicConfig(
    filename="output/log_file.log",
    encoding="utf-8",
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)

logger.debug("===============  Started!  ===============")


# directory = "../CAP6610SP24_training_set"
directory = "../toy_training_set"

train_list = SongList(directory)
train_list.save_feature()

# # Get the first song and plot features
# song = train_list.song_list[0]
# print(song)
# song.get_all_features()
# song.plot_all_features()

# # Split song into 10-second segments
# song.split_song()
# seg = song.segments[0]
# seg.get_all_features()
# seg.plot_all_features()
