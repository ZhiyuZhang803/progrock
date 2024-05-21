from Song import Song
from concurrent.futures import ThreadPoolExecutor

import os
import json
import logging

logger = logging.getLogger(__name__)

problematic_songs = []


class SongList:
    def __init__(self, directory):
        """
        eg., directory = "../CAP6610SP24_training_set"

        """
        self.directory = directory
        self.song_list = []
        self.prog_count = 0
        self.nonprog_count = 0

        logger.debug(f"Generating song list from {directory} ...")
        self.read_folder()

    def read_folder(self):
        files_list = {}
        with ThreadPoolExecutor() as executor:
            song_tasks = []
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
                            # self.add_song(song_path, label)
                            song_tasks.append(
                                executor.submit(self.add_song, song_path, label)
                            )
                            logger.debug(song_path)
                        else:
                            logger.error(song_path)

                if song_count > 0 or not_song_files:  # folder not empty
                    content["songs"] = song_count
                    content["others"] = not_song_files
                    files_list[dir_name] = content

        for task in song_tasks:
            task.result()  # Wait for completion

        logger.debug(files_list)
        logger.info(self)

    def add_song(self, path, label):
        new_song = Song(path, label)
        self.song_list.append(new_song)
        if label:
            self.prog_count += 1
        else:
            self.nonprog_count += 1

    def save_feature(self):
        self.feature = {
            "song_name_arr": [],
            "segment_i_arr": [],
            "audio_arr": [],
            "spectro_arr": [],
            "chroma_arr": [],
            "onset_arr": [],
            "MFCC_arr": [],
            "label_arr": [],
        }

        for song in self.song_list:
            song.process_song()
            for segment in song.segments:
                self.feature["song_name_arr"].append(song.name)
                self.feature["segment_i_arr"].append(segment.name)
                self.feature["label_arr"].append(song.label)
                self.feature["audio_arr"].append(segment.audio.tolist())
                self.feature["spectro_arr"].append(segment.spectro.tolist())
                self.feature["chroma_arr"].append(segment.chroma.tolist())
                self.feature["onset_arr"].append(segment.onset.tolist())
                self.feature["MFCC_arr"].append(segment.MFCC.tolist())

        with open("../data/feature_data.json", "w") as json_file:
            json.dump(self.feature, json_file, indent=4)

    def __str__(self) -> str:
        return f"Song List From {self.directory}:\n\
                \tProgrock: {self.prog_count} songs\n\
                \tNonProgrock: {self.nonprog_count} songs\n\
                \tTotal: {len(self.song_list)} songs"
