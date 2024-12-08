from pathlib import Path
from src.datasets.base_dataset import BaseDataset
from pathlib import Path
from torch.utils.data import Dataset


class CustomTTSDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.transcription_dir = self.data_dir / "transcriptions"
        if not self.transcription_dir.exists():
            raise FileNotFoundError(f"Directory {self.transcription_dir} does not exist.")

        self.transcription_files = list(self.transcription_dir.glob("*.txt"))
        if len(self.transcription_files) == 0:
            raise ValueError(f"No transcription files found in {self.transcription_dir}")

        self.items = []
        for txt_file in self.transcription_files:
            utterance_id = txt_file.stem
            self.items.append((utterance_id, txt_file))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        utterance_id, txt_file = self.items[idx]
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return {"utterance_id": utterance_id, "text": text}
