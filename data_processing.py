import os


def check_folder(directory):
    files_list = {}
    for root, dirs, files in os.walk(directory):
        dir_name = os.path.relpath(root, directory)
        content = {}
        song_count = 0
        not_song_files = []
        for file in files:
            if not file.endswith(".mp3"):
                not_song_files.append(file)
            else:
                song_count += 1
        if song_count > 0 or not_song_files:  # folder not empty
            content["songs"] = song_count
            content["others"] = not_song_files
            files_list[dir_name] = content
    print(files_list)
    return files_list


# Example usage
train_set_folder = "../CAP6610SP24_training_set"
file_list = check_folder(train_set_folder)
