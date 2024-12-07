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

    if 'full_waveform' in dataset_items[0]:
        full_wavs = [item['full_waveform'] for item in dataset_items]
        max_full_wavs_length = max(fw.size(-1) for fw in full_wavs)
        padded_full_waveforms = []
        for fw in full_wavs:
            length_diff = max_full_wavs_length - fw.size(-1)
            fw_padded = F.pad(fw, (0, length_diff))
            padded_full_waveforms.append(fw_padded)
        batch_full_waveforms = torch.stack(padded_full_waveforms)
    else:
        batch_full_waveforms = None

    if 'full_mel' in dataset_items[0]:
        full_mels = [item['full_mel'] for item in dataset_items]
        max_full_mel_length = max(fm.size(-1) for fm in full_mels)
        padded_full_mels = []
        for fm in full_mels:
            length_diff = max_full_mel_length - fm.size(-1)
            fm_padded = F.pad(fm, (0, length_diff))
            padded_full_mels.append(fm_padded)
        batch_full_mels = torch.stack(padded_full_mels)
    else:
        batch_full_mels = None

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

    result = {
        'waveform': batch_waveforms,
        'mel_spec': batch_mel_specs
    }

    if batch_full_waveforms is not None:
        result['full_waveform'] = batch_full_waveforms
    if batch_full_mels is not None:
        result['full_mel'] = batch_full_mels

    return result
