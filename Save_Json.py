import json
import os
import time

from Save_Signal import load_and_segment_song, transfer_features


PROG_PATH = "CAP6610SP24_training_set/Progressive_Rock_Songs"
NON_PROG_OTHER = "CAP6610SP24_training_set/Not_Progressive_Rock/Other_Songs"
NON_PROG_TOP_POPS = "CAP6610SP24_training_set/Not_Progressive_Rock/Top_Of_The_Pops"
JSON_PATH = "Feature_Extraction.json"

json_data = {
    "Name": [],
    "Snippets": [],
    "Label":[],
    "MFCC":[],
    "Spectrogram":[],
    "Chromagram":[],
    "Onset":[]
}

empty_list = []

def write_json(input_path, song_type_label):
    """
    Load a song, extract features, and write to json file


    Parameters:
        input_path (str): The folder that stores certain type of music.
        song_type_label (obj_any): The label given to the songs from the input path.

    No Returns, just write information to the json_data.

    Notice:
        Features are extracted with default parameters. 10-sec snippets.
        When Librosa fails to load the file, store in an empty_list.
    """
    # List all files under certain path
    song_list = os.listdir(input_path)

    # Start time
    start_time = time.time()

    # Iterate through all possible songs
    for i in song_list:
        full_path = input_path + "/" + i
        segments = load_and_segment_song(full_path)
        # Sometimes librosa returns empty
        if segments == []:
            empty_list.append(i)
            break
        # Iterate through all possible segments
        for snippet in range(len(segments)):
            array_signal = segments[snippet]
            M_DB, C, onset_env, MFCCs = transfer_features(array_signal)
            json_data["Name"].append(i)
            json_data["Snippets"].append(snippet+1) # Start from 1
            json_data["Label"].append(song_type_label)
            json_data["MFCC"].append(MFCCs)
            json_data["Spectrogram"].append(M_DB)
            json_data["Chromagram"].append(C)
            json_data["Onset"].append(onset_env)
        current_time = time.time()
        print("Finish: ", i, " Total Time (s): ", round(current_time-start_time))


if __name__ == '__main__':
    write_json(PROG_PATH, 0)
    write_json(NON_PROG_OTHER, 1)
    write_json(NON_PROG_TOP_POPS, 2)

    with open(JSON_PATH, "w") as fp:
        json.dump(json_data, fp, indent=4)

    print(empty_list)