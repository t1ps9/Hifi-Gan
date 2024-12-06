import torch
import torch.nn.functional as F


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    wavs = [item['waveform'] for item in dataset_items]
    mel_specs = [item['mel_spec'] for item in dataset_items]

    max_wavs_length = max(waveform.size(-1) for waveform in wavs)
    max_mel_spec_length = max(mel_spec.size(-1) for mel_spec in mel_specs)

    padded_waveforms = []
    for wav in wavs:
        length_diff = max_wavs_length - wav.size(-1)
        wav_padded = F.pad(wav, (0, length_diff))
        padded_waveforms.append(wav_padded)
    batch_waveforms = torch.stack(padded_waveforms)

    padded_mels = []
    for mel in mel_specs:
        length_diff = max_mel_spec_length - mel.size(-1)
        mel_padded = F.pad(mel, (0, length_diff))
        padded_mels.append(mel_padded)
    batch_mel_specs = torch.stack(padded_mels)

    return {
        'waveform': batch_waveforms,
        'mel_spec': batch_mel_specs
    }
