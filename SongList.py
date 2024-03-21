from Song import Song
import os
import logging

logger = logging.getLogger(__name__)

problematic_songs = [
    "11-zao-a_last_time_for_everything.mp3",
    "Green Day -07- Basket Case.mp3",
    "1979_The_Knack_My_Sharona.mp3",
]


class SongList:
    def __init__(self, directory):
        """
        directory = "../CAP6610SP24_training_set"

        """
        self.directory = directory
        self.song_list = []
        self.prog_count = 0
        self.nonprog_count = 0

        logger.debug(f"Generating song list from {directory} ...")
        self.read_folder()

    def read_folder(self):
        files_list = {}
        for root, _, files in os.walk(self.directory):
            dir_name = os.path.relpath(root, self.directory)
            label = 0 if "Not" in dir_name else 1
            content = {}
            song_count = 0
            not_song_files = []
            for file in files:
                if not file.endswith(".mp3"):
                    not_song_files.append(file)
                else:
                    song_count += 1
                    song_path = os.path.join(root, file)
                    if file not in problematic_songs:
                        self.add_song(song_path, label)
                        logger.debug(song_path)
                    else:
                        logger.error(song_path)

            if song_count > 0 or not_song_files:  # folder not empty
                content["songs"] = song_count
                content["others"] = not_song_files
                files_list[dir_name] = content

        logger.debug(files_list)
        logger.info(self)

    def add_song(self, path, label):
        new_song = Song(path, label)
        self.song_list.append(new_song)
        if label:
            self.prog_count += 1
        else:
            self.nonprog_count += 1

    def __str__(self) -> str:
        return (f"Song List From {self.directory}:\n\
                \tProgrock: {self.prog_count} songs\n\
                \tNonProgrock: {self.nonprog_count} songs\n\
                \tTotal: {len(self.song_list)} songs")
