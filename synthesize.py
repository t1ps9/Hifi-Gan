from tacotron2.text import text_to_sequence
import torch
import torchaudio
import hydra
from omegaconf import DictConfig
from pathlib import Path
from torch.utils.data import DataLoader
import sys
from hydra.utils import instantiate
from src.datasets.custom_tts_dataset import CustomTTSDataset
import warnings
warnings.filterwarnings("ignore")

_orig_torch_load = torch.load


def torch_load_cpu(*args, **kwargs):
    kwargs['map_location'] = 'cpu'
    return _orig_torch_load(*args, **kwargs)


torch.load = torch_load_cpu


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(config.inferencer.from_pretrained, map_location="cpu")
    vocoder = instantiate(config.model).to(device)
    vocoder.load_state_dict(checkpoint["state_dict"])
    vocoder.eval()

    tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
    tacotron2.to(device).eval()

    output_dir = Path(config.inferencer.save_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    def text_to_mel(text: str):
        sequence = text_to_sequence(text, ['english_cleaners'])
        sequence = torch.IntTensor(sequence)[None, :].to(device)
        input_lengths = torch.IntTensor([sequence.size(1)]).to(device)

        with torch.no_grad():
            mel, _, _ = tacotron2.infer(sequence, input_lengths)
        return mel

    if config.inferencer.text_query and len(config.inferencer.text_query.strip()) > 0:
        text = config.inferencer.text_query.strip()
        mel = text_to_mel(text)
        mel = mel.to(device)
        with torch.no_grad():
            audio = vocoder.generator(mel)
        audio = audio.squeeze(0)
        torchaudio.save(str(output_dir / "inference_from_text_query.wav"), audio.cpu(), 22050)
        print("Inference result saved to:", output_dir / "inference_from_text_query.wav")
    else:
        dataset = CustomTTSDataset(config.datasets.test.audio_dir)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        for batch in dataloader:
            utterance_id = batch["utterance_id"][0]
            text = batch["text"][0]
            mel = text_to_mel(text)
            mel = mel.to(device)
            with torch.no_grad():
                audio = vocoder.generator(mel)
            audio = audio.squeeze(0)
            out_path = output_dir / f"{utterance_id}.wav"
            torchaudio.save(str(out_path), audio.cpu(), 22050)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
