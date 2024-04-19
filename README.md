# Prog-rock vs. Everything Else

An ML algorithm to distinguish Progressive Rock music from everything else.

## Extract and save feature

Run `save_feature.py`.

- Each subfolder in the `tran_set\` has an `.ods` file stating the list of songs, and `.mp3` files for songs.
    - `Progressive_Rock_Songs\`: {'songs': 142, 'others': ['prog_train.ods']}
    - `Not_Progressive_Rock\Top_Of_The_Pops\`: {'songs': 87, 'others': ['notprog_top_pops_train.ods']}
    - `Not_Progressive_Rock\Other_Songs\`: {'songs': 272, 'others': ['notprog_other_train.ods']}

- Load music files using `librosa`.
    - It has [features](https://librosa.org/doc/latest/feature.html) of 
        - Spectral
            - Chromagram
            - Mel-scaled spectrogram
            - Mel-frequency cepstral coefficients (MFCCs)
        - Rythym
            - Tempo
        - Others


## Run algorithm

- Place saved feature `json` files in relative path to this repo at `../data/[feature.json]`. The default paths are set in `train.py` as:
    ```python
    non_prog_other_path = "../data/Feature_Extraction_Other.json"
    non_prog_pop_path = "../data/Feature_Extraction_Top_Pop.json"
    prog_path = "../data/Feature_Extraction_Prog.json"
    ```

- Run `main.py`.

- For new CNN models, just import them into `main.py` and add a line to run `algo([model_object()], [model_name_string])`.


## Output

- Feature plots are saved in `output` folder as `.pdf`.

- Model results are saved in `output/model` folder named after the model name. These include
    - Confusion matrices for train/test snippets/songs (all in one file)
    - Model test labeling result
    - Model pickle file

- Log file is generated at `output/log_file.log`