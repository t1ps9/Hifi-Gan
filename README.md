## How To Use

Clone repository

```bash
git clone https://github.com/t1ps9/dla_hw1_asr
```
Move to folder

```bash
cd dla_hw1_asr
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
python download_model_weights.py
```

Run inference

```bash
python inference.py datasets.test.audio_dir='Path to wav'
```

Calc wer and cer

```bash
python calc_wer_and_cer.py --target_dir <Path to transcriptions (dir)>
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
