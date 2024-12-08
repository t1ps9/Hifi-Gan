## How To Use

Clone repository

```bash
git clone https://github.com/t1ps9/Hifi-Gan
```
Move to folder

```bash
cd Hifi-Gan
```

Create and activate env

```bash
conda create -n tmp python=3.11

conda activate tmp
```

Install requirements

```bash
pip install -r requirements.txt
```

Dowload model weights

```bash
python3 download_model_weights.py
```

Run inference with CustomDirDataset 

```bash
python3 synthesize.py datasets.test.audio_dir=<Path to folder>
```

You can check the inference on 5 texts that were given here https://github.com/markovka17/dla/tree/2024/hw3_nv

```bash
python3 synthesize.py datasets.test.audio_dir=test_tts
```

Run inference with text query  

```bash
python3 synthesize.py inferencer.text_query="WRITE THE TEXT HERE"
```
Predict wavs is saved to the path specified during output:

Saved...


## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
