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

## Models

We build several baseline models based on the last years' best project, as well as several new models.

### 1 Baseline
conv1d(43,64) -> conv1d(64,64) -> linear(6848,600) -> linear(600,10) -> linear(10,2)

### 2 ModifiedBaseline
conv1d(43,64) -> conv1d(64,128) -> linear(13696,600) -> linear(600,30) -> linear(30,4) -> linear(4,2)

### 3 DeepBaseline
conv1d(43,64) -> conv1d(64,128) -> conv1d(128,256) -> conv1d(256,512) -> linear(13312,100) -> linear(100,10) -> linear(10,2)

### 4 DeepWiseBaseline
conv1d(43,86) -> conv1d(86,172) -> conv1d(172,344) -> linear(18232,2)

### 5 AcousticModel (O’Brien 2016)
conv1d(43,64) -> conv1d(64,128) -> conv1d(128,128) -> conv1d(128,64) -> conv1d(64,32)-> conv1d(32,32) -> linear(832,2)
- O’Brien, Tim. "Musical Structure Segmentation with Convolutional Neural Networks." 17th International Society for Music Information Retrieval Conference. 2016.

### 6 GenreModel 
conv1d(43,128) -> conv1d(128,128) -> conv1d(128,256) -> conv1d(256,256) -> conv1d(256,256) -> conv1d(256,256) -> conv1d(256,512)-> conv1d(512,10) -> linear(40,2)

### 7 ResnetModel (Allamy and Alessandro 2021)
conv1d(43,128) -> res1d(128,128) -> res1d(128,256) -> … -> res1d(256,512)-> conv1d(512,10) -> linear(40,2)
- Allamy, Safaa, and Alessandro Lameiras Koerich. "1D CNN architectures for music genre classification." 2021 IEEE symposium series on computational intelligence (SSCI). IEEE, 2021.



## Run algorithm

- Place saved feature `json` files in relative path to this repo at `../data/[feature.json]`. The default paths are set in `train.py` as:
    ```python
    non_prog_other_path = "../data/Feature_Extraction_Other.json"
    non_prog_pop_path = "../data/Feature_Extraction_Top_Pop.json"
    prog_path = "../data/Feature_Extraction_Prog.json"
    ```

- Run `main.py`.

- For new CNN models, just import them into `main.py` and add corresponding `(model_name, model())` to `model_dict` in `main.py`.


## Output

- Feature plots are saved in `output` folder as `.pdf`.

- Model results are saved in `output/model` folder named after the model name. These include
    - Confusion matrices for train/test snippets/songs (all in one file)
    - Average confusion matrix after multiple runs
    - Model test labeling result
    - Model pickle file

- Log file is generated at `output/log_file.log`