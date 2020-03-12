from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import audio
from nnmnkwii.datasets import vctk
from nnmnkwii.io import hts
from hparams import hparams
from os.path import exists
import librosa


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    speakers = vctk.available_speakers

    td = vctk.TranscriptionDataSource(in_dir, speakers=speakers)
    transcriptions = td.collect_files()
    speaker_ids = td.labels
    wav_paths = vctk.WavFileDataSource(
        in_dir, speakers=speakers).collect_files()

    for index, (speaker_id, text, wav_path) in enumerate(
            zip(speaker_ids, transcriptions, wav_paths)):
        futures.append(executor.submit(
            partial(_process_utterance, out_dir, index + 1, speaker_id, wav_path, text)))
    return [future.result() for future in tqdm(futures)]


def start_at(labels):
    has_silence = labels[0][-1] == "pau"
    if not has_silence:
        return labels[0][0]
    for i in range(1, len(labels)):
        if labels[i][-1] != "pau":
            return labels[i][0]
    assert False


def end_at(labels):
    has_silence = labels[-1][-1] == "pau"
    if not has_silence:
        return labels[-1][1]
    for i in range(len(labels) - 2, 0, -1):
        if labels[i][-1] != "pau":
            return labels[i][1]
    assert False


def _process_utterance(out_dir, index, speaker_id, wav_path, text):
    sr = hparams.sample_rate

    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)

    lab_path = wav_path.replace("wav48\\", "lab\\").replace(".wav", ".lab")

    # Trim silence from hts labels if available
    if exists(lab_path):
        labels = hts.load(lab_path)
        b = int(start_at(labels) * 1e-7 * sr)
        e = int(end_at(labels) * 1e-7 * sr)
        wav = wav[b:e]
        wav, _ = librosa.effects.trim(wav, top_db=25)
    else:
        wav, _ = librosa.effects.trim(wav, top_db=15)

    if hparams.rescaling:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

    # world parameters
    f0, sp, ap = audio.world(wav, hparams.sample_rate)
    f0 = (f0 / hparams.f0_norm).astype(np.float32)
    sp = audio._normalize(sp).astype(np.float32)
    ap = ap.astype(np.float32)
    world_frames = f0.shape[0]

    # Write the spectrograms to disk:
    spectrogram_filename = 'vctk-spec-%05d.npy' % index
    mel_filename = 'vctk-mel-%05d.npy' % index
    f0_filename = 'vctk-f0-%05d.npy' % index
    sp_filename = 'vctk-sp-%05d.npy' % index
    ap_filename = 'vctk-ap-%05d.npy' % index
    np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, f0_filename), f0, allow_pickle=False)
    np.save(os.path.join(out_dir, sp_filename), sp, allow_pickle=False)
    np.save(os.path.join(out_dir, ap_filename), ap, allow_pickle=False)

    # Return a tuple describing this training example:
    return (spectrogram_filename, mel_filename, n_frames,  f0_filename, sp_filename,
            ap_filename, world_frames, text, speaker_id)
