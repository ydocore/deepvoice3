"""Test Dataloader time.

usage: training_module.py [options]

options:
    --data-root=<dir>            Directory contains preprocessed features.
    --checkpoint-dir=<dir>       Directory where to save model checkpoints [default: checkpoints].
    --hparams=<parmas>           Hyper parameters [default: ].
    --preset=<json>              Path of preset parameters (json).
    --checkpoint=<path>          Restore model from checkpoint path if given.
    --checkpoint-seq2seq=<path>  Restore seq2seq model from checkpoint path.
    --checkpoint-postnet=<path>  Restore postnet model from checkpoint path.
    --restore-parts=<path>       Restore part of the model.
    --log-event-path=<name>      Log event path.
    --reset-optimizer            Reset optimizer.
    --load-embedding=<path>      Load embedding from checkpoint.
    --speaker-id=<N>             Use specific speaker of data in case for multi-speaker datasets.
    -h, --help                   Show this help message and exit
"""
from docopt import docopt
import sys
import gc
import platform
from os.path import dirname, join
from tqdm import tqdm, trange
from datetime import datetime

# The deepvoice3 model
from deepvoice3_pytorch import frontend, builder
import audio
import lrschedule

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
from torch.utils.data.sampler import Sampler
import numpy as np

from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from os.path import join, expanduser
import random

import librosa.display
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys
import os
from matplotlib import cm
from warnings import warn
from hparams import hparams, hparams_debug_string
import time

use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False

_frontend = getattr(frontend, hparams.frontend)

def _pad(seq, max_len, constant_values=0):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=constant_values)


def _pad_2d(x, max_len, b_pad=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
               mode="constant", constant_values=0)
    return x


def plot_alignment(alignment, path, info=None):
    fig, ax = plt.subplots()
    im = ax.imshow(
        alignment,
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.savefig(path, format='png')
    plt.close()


class TextDataSource(FileDataSource):
    def __init__(self, data_root, speaker_id=None):
        self.data_root = data_root
        self.speaker_ids = None
        self.multi_speaker = False
        # If not None, filter by speaker_id
        self.speaker_id = speaker_id

    def collect_files(self):
        meta = join(self.data_root, "train.txt") 
        with open(meta, "rb") as f:
            lines = f.readlines()
        l = lines[0].decode("utf-8").split("|")
        assert len(l) == 8 or len(l) == 9
        self.multi_speaker = len(l) == 9
        texts = list(map(lambda l: l.decode("utf-8").split("|")[7], lines))
        if self.multi_speaker:
            speaker_ids = list(map(lambda l: int(l.decode("utf-8").split("|")[-1]), lines))
            # Filter by speaker_id
            # using multi-speaker dataset as a single speaker dataset
            if self.speaker_id is not None:
                indices = np.array(speaker_ids) == self.speaker_id
                texts = list(np.array(texts)[indices])
                self.multi_speaker = False
                return texts

            return texts, speaker_ids
        else:
            return texts

    def collect_features(self, *args):
        if self.multi_speaker:
            text, speaker_id = args
        else:
            text = args[0]
        global _frontend
        if _frontend is None:
            _frontend = getattr(frontend, hparams.frontend)
        seq = _frontend.text_to_sequence(text, p=hparams.replace_pronunciation_prob)

        if platform.system() == "Windows":
            if hasattr(hparams, 'gc_probability'):
                _frontend = None  # memory leaking prevention in Windows
                if np.random.rand() < hparams.gc_probability:
                    gc.collect()  # garbage collection enforced
                    print("GC done")

        if self.multi_speaker:
            return np.asarray(seq, dtype=np.int32), int(speaker_id)
        else:
            return np.asarray(seq, dtype=np.int32)


class _NPYDataSource(FileDataSource):
    def __init__(self, data_root, col, speaker_id=None):
        self.data_root = data_root
        self.col = col
        self.frame_lengths = []
        self.world_lengths = []
        self.speaker_id = speaker_id

    def collect_files(self):
        meta = join(self.data_root, "train.txt")
        with open(meta, "rb") as f:
            lines = f.readlines()
        l = lines[0].decode("utf-8").split("|")
        assert len(l) == 8 or len(l) == 9
        multi_speaker = len(l) == 9
        self.frame_lengths = list(
            map(lambda l: int(l.decode("utf-8").split("|")[2]), lines))
        self.world_lengths = list(
            map(lambda l: int(l.decode("utf-8").split("|")[6]), lines))

        paths = list(map(lambda l: l.decode("utf-8").split("|")[self.col], lines))
        paths = list(map(lambda f: join(self.data_root, f), paths))

        if multi_speaker and self.speaker_id is not None:
            speaker_ids = list(map(lambda l: int(l.decode("utf-8").split("|")[-1]), lines))
            # Filter by speaker_id
            # using multi-speaker dataset as a single speaker dataset
            indices = np.array(speaker_ids) == self.speaker_id
            paths = list(np.array(paths)[indices])
            self.frame_lengths = list(np.array(self.frame_lengths)[indices])
            # aha, need to cast numpy.int64 to int
            self.frame_lengths = list(map(int, self.frame_lengths))

        return paths

    def collect_features(self, path):
        return np.load(path)


class MelSpecDataSource(_NPYDataSource):
    def __init__(self, data_root, speaker_id=None):
        super(MelSpecDataSource, self).__init__(data_root, 1, speaker_id)


class LinearSpecDataSource(_NPYDataSource):
    def __init__(self, data_root, speaker_id=None):
        super(LinearSpecDataSource, self).__init__(data_root, 0, speaker_id)

class F0DataSource(_NPYDataSource):
    def __init__(self, data_root, speaker_id=None):
        super(F0DataSource, self).__init__(data_root, 3, speaker_id)

class SpDataSource(_NPYDataSource):
    def __init__(self, data_root, speaker_id=None):
        super(SpDataSource, self).__init__(data_root, 4, speaker_id)


class ApDataSource(_NPYDataSource):
    def __init__(self, data_root, speaker_id=None):
        super(ApDataSource, self).__init__(data_root, 5, speaker_id)


class PartialyRandomizedSimilarTimeLengthSampler(Sampler):
    """Partially randmoized sampler

    1. Sort by lengths
    2. Pick a small patch and randomize it
    3. Permutate mini-batchs
    """

    def __init__(self, lengths, batch_size=16, batch_group_size=None,
                 permutate=True):
        self.lengths, self.sorted_indices = torch.sort(torch.LongTensor(lengths))
        self.batch_size = batch_size
        if batch_group_size is None:
            batch_group_size = min(batch_size * 32, len(self.lengths))
            if batch_group_size % batch_size != 0:
                batch_group_size -= batch_group_size % batch_size

        self.batch_group_size = batch_group_size
        assert batch_group_size % batch_size == 0
        self.permutate = permutate

    def __iter__(self):
        indices = self.sorted_indices.clone()
        batch_group_size = self.batch_group_size
        s, e = 0, 0
        for i in range(len(indices) // batch_group_size):
            s = i * batch_group_size
            e = s + batch_group_size
            random.shuffle(indices[s:e])

        # Permutate batches
        if self.permutate:
            perm = np.arange(len(indices[:e]) // self.batch_size)
            random.shuffle(perm)
            indices[:e] = indices[:e].view(-1, self.batch_size)[perm, :].view(-1)

        # Handle last elements
        s += batch_group_size
        if s < len(indices):
            random.shuffle(indices[s:])

        return iter(indices)

    def __len__(self):
        return len(self.sorted_indices)





def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')


def save_alignment(path, attn, global_step):
    plot_alignment(attn.T, path, info="{}, {}, step={}".format(
        hparams.builder, time_string(), global_step))


def prepare_spec_image(spectrogram):
    # [0, 1]
    spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
    spectrogram = np.flip(spectrogram, axis=1)  # flip against freq axis
    return np.uint8(cm.magma(spectrogram.T) * 255)


def eval_model(global_step, writer, device, model, checkpoint_dir, ismultispeaker):
    # harded coded
    texts = [
        "And debtors might practically have as much as they liked%if they could only pay for it.",
        "There's a way to measure the acute emotional intelligence that has never gone out of style.",
        "President trump met with other leaders at the group of 20 conference.",
        "Generative adversarial network or variational auto encoder.",
        "Please call stella.",
        "Some have accepted this as a miracle without any physical explanation.",
    ]
    import synthesis
    synthesis._frontend = _frontend

    eval_output_dir = join(checkpoint_dir, "eval")
    os.makedirs(eval_output_dir, exist_ok=True)

    # Prepare model for evaluation
    model_eval = build_model(training_type=model.training_type).to(device)
    model_eval.load_state_dict(model.state_dict())

    # hard coded
    speaker_ids = [0, 1, 10] if ismultispeaker else [None]
    for speaker_id in speaker_ids:
        speaker_str = "multispeaker{}".format(speaker_id) if speaker_id is not None else "single"

        for idx, text in enumerate(texts, 1):
            signal, alignments, _, mel = synthesis.tts(
                model_eval, text, p=1, speaker_id=speaker_id, fast=True)
            signal /= np.max(np.abs(signal))

            # Alignment
            for i, alignment in enumerate(alignments, 1):
                alignment_dir = join(eval_output_dir, "alignment_layer{}".format(i))
                os.makedirs(alignment_dir, exist_ok=True)
                path = join(alignment_dir, "step{:09d}_text{}_{}_layer{}_alignment.png".format(
                    global_step, idx, speaker_str, i))
                save_alignment(path, alignment, global_step)
                tag = "eval_text_{}_alignment_layer{}_{}".format(idx, i, speaker_str)
                writer.add_image(tag, np.uint8(cm.viridis(np.flip(alignment, 1)) * 255).T, global_step)

            # Mel
            writer.add_image("(Eval) Predicted mel spectrogram text{}_{}".format(idx, speaker_str),
                             prepare_spec_image(mel).transpose(2,0,1), global_step)

            # Audio
            path = join(eval_output_dir, "step{:09d}_text{}_{}_predicted.wav".format(
                global_step, idx, speaker_str))
            audio.save_wav(signal, path)

            try:
                writer.add_audio("(Eval) Predicted audio signal {}_{}".format(idx, speaker_str),
                                 signal, global_step, sample_rate=hparams.sample_rate)
            except Exception as e:
                warn(str(e))
                pass


def save_states(global_step, writer, mel_outputs, converter_outputs, attn, mel, y,
                input_lengths, checkpoint_dir=None):

    def save_world(tuple_outputs, save_str, global_step=global_step):
        _, tar_f0, tar_sp, tar_ap = tuple_outputs
        fig = plt.figure()
        f0 = tar_f0[idx].cpu().data.numpy() * 400
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(f0)
        save_f0 = save_str + ' f0'
        writer.add_figure(save_f0, fig, global_step)

        # sp save
        sp = tar_sp[idx].cpu().data.numpy()
        s = prepare_spec_image(sp)
        save_sp = save_str + ' sp'
        writer.add_image(save_sp, s.transpose(2, 0, 1), global_step)

        # ap save
        ap = tar_ap[idx].cpu().data.numpy()
        a = prepare_spec_image(ap)
        save_ap = save_str + ' ap'
        writer.add_image(save_ap, a.transpose(2, 0, 1), global_step)

    print("Save intermediate states at step {}".format(global_step))

    # idx = np.random.randint(0, len(input_lengths))
    idx = min(1, len(input_lengths) - 1)
    input_length = input_lengths[idx]

    # Alignment
    # Multi-hop attention
    if attn is not None and attn.dim() == 4:
        for i, alignment in enumerate(attn):
            alignment = alignment[idx].cpu().data.numpy()
            tag = "alignment_layer{}".format(i + 1)
            writer.add_image(tag, np.uint8(cm.viridis(np.flip(alignment, 1)) * 255).T, global_step) #転置消去

            # save files as well for now
            alignment_dir = join(checkpoint_dir, "alignment_layer{}".format(i + 1))
            os.makedirs(alignment_dir, exist_ok=True)
            path = join(alignment_dir, "step{:09d}_layer_{}_alignment.png".format(
                global_step, i + 1))
            save_alignment(path, alignment, global_step)

    # Predicted mel spectrogram
    if mel_outputs is not None:
        mel_output = mel_outputs[idx].cpu().data.numpy()
        mel_output = prepare_spec_image(audio._denormalize(mel_output))
        writer.add_image("Predicted mel spectrogram", mel_output.transpose(2,0,1), global_step)

        #target
        mel_output = mel[idx].cpu().data.numpy()
        mel_output = prepare_spec_image(audio._denormalize(mel_output))
        writer.add_image("Target mel spectrogram", mel_output.transpose(2, 0, 1), global_step)

    if converter_outputs is not None:
        # Predicted world parameter
        if type(converter_outputs) is tuple:
            #save predicted
            save_world(converter_outputs, 'Predicted')

            #save target
            save_world(y, 'Target')

            #save world signal
            _, f0s, sps, aps = converter_outputs
            f0 = f0s[idx].cpu().data.numpy() * 400
            sp = sps[idx].cpu().data.numpy()
            ap = aps[idx].cpu().data.numpy()
            # world vocoder
            signal = audio.world_synthesize(f0, sp, ap)
            signal /= np.max(np.abs(signal))
        # Predicted spectrogram
        else:
            linear_output = converter_outputs[idx].cpu().data.numpy()
            spectrogram = prepare_spec_image(audio._denormalize(linear_output))
            writer.add_image("Predicted linear spectrogram", spectrogram.transpose(2,0,1), global_step)

            # Predicted audio signal
            signal = audio.inv_spectrogram(linear_output.T)
            signal /= np.max(np.abs(signal))
            path = join(checkpoint_dir, "step{:09d}_predicted.wav".format(
                global_step))

            #target
            linear_output = y[idx].cpu().data.numpy()
            spectrogram = prepare_spec_image(audio._denormalize(linear_output))
            writer.add_image("Target linear spectrogram", spectrogram.transpose(2, 0, 1), global_step)

        try:
            writer.add_audio("Predicted audio signal", signal, global_step, sample_rate=hparams.sample_rate)
        except Exception as e:
            warn(str(e))
            pass
        audio.save_wav(signal, path)


def logit(x, eps=1e-8):
    return torch.log(x + eps) - torch.log(1 - x + eps)


def save_checkpoint(model, optimizer, global_step, checkpoint_dir, epoch):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}_{}.pth".format(global_step, model.training_type))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": global_step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer):
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model, global_step, global_epoch


def _load_embedding(path, model):
    state = _load(path)["state_dict"]
    key = "seq2seq.encoder.embed_tokens.weight"
    model.seq2seq.encoder.embed_tokens.weight.data = state[key]

# https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3


def restore_parts(path, model):
    print("Restore part of the model from: {}".format(path))
    state = _load(path)["state_dict"]
    model_dict = model.state_dict()
    valid_state_dict = {k: v for k, v in state.items() if k in model_dict}

    try:
        model_dict.update(valid_state_dict)
        model.load_state_dict(model_dict)
    except RuntimeError as e:
        # there should be invalid size of weight(s), so load them per parameter
        print(str(e))
        model_dict = model.state_dict()
        for k, v in valid_state_dict.items():
            model_dict[k] = v
            try:
                model.load_state_dict(model_dict)
            except RuntimeError as e:
                print(str(e))
                warn("{}: may contain invalid size of weight. skipping...".format(k))

def build_model(training_type='seq2seq'): # linear or WORLD or WaveNet
    # Call the "hparams.builder" function in builder
    model = getattr(builder, hparams.builder)(
        n_speakers=hparams.n_speakers,
        speaker_embed_dim=hparams.speaker_embed_dim,
        n_vocab=_frontend.n_vocab,
        embed_dim=hparams.text_embed_dim,
        mel_dim=hparams.num_mels,
        linear_dim=hparams.fft_size // 2 + 1,
        r=hparams.outputs_per_step,
        padding_idx=hparams.padding_idx,
        dropout=hparams.dropout,
        kernel_size=hparams.kernel_size,
        encoder_channels=hparams.encoder_channels,
        num_encoder_layer=hparams.num_encoder_layer,
        decoder_channels=hparams.decoder_channels,
        num_decoder_layer=hparams.num_decoder_layer,
        attention_hidden=hparams.attention_hidden,
        converter_channels=hparams.converter_channels,
        num_converter_layer=hparams.num_converter_layer,
        query_position_rate=hparams.query_position_rate,
        key_position_rate=hparams.key_position_rate,
        position_weight=hparams.position_weight,
        use_memory_mask=hparams.use_memory_mask,
        trainable_positional_encodings=hparams.trainable_positional_encodings,
        force_monotonic_attention=hparams.force_monotonic_attention,
        use_decoder_state_for_postnet_input=hparams.use_decoder_state_for_postnet_input,
        max_positions=hparams.max_positions,
        speaker_embedding_weight_std=hparams.speaker_embedding_weight_std,
        freeze_embedding=hparams.freeze_embedding,
        window_ahead=hparams.window_ahead,
        window_backward=hparams.window_backward,
        world_upsample=hparams.world_upsample,
        sp_fft_size=hparams.sp_fft_size,
        training_type=training_type
    )
    return model


'''
#以下，データ読み込みのテスト用
class PyTorchDataset(object):
    def __init__(self, X, Mel, Y, F0, SP, AP):
        self.X = X
        self.Mel = Mel
        self.Y = Y
        self.F0 = F0
        self.SP = SP
        self.AP = AP
        # alias
        self.multi_speaker = X.file_data_source.multi_speaker

    def __getitem__(self, idx):
        if self.multi_speaker:
            text, speaker_id = self.X[idx]
            return text, self.Mel[idx], self.Y[idx], self.F0[idx], self.SP[idx], self.AP[idx], speaker_id
        else:
            return self.X[idx], self.Mel[idx], self.Y[idx], self.F0[idx], self.SP[idx], self.AP[idx]


    def __len__(self):
        return len(self.X)


def collate_fn(batch):
    """Create batch"""
    r = hparams.outputs_per_step
    multi_speaker = len(batch[0]) == 7

    # Lengths
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = max(input_lengths)

    target_lengths = [len(x[1]) for x in batch]
    max_target_len = max(target_lengths)

    world_lengths = [len(x[3]) for x in batch]

    if max_input_len % r != 0:
        max_input_len += r - max_input_len % r
        assert max_input_len % r == 0

    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0

    # Set 0 for zero beginning padding
    # imitates initial decoder states
    b_pad = r
    max_target_len += b_pad
    max_world_len = int(max_target_len * hparams.world_upsample)

    a = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.int)
    x_batch = torch.LongTensor(a)

    input_lengths = torch.LongTensor(input_lengths)
    target_lengths = torch.LongTensor(target_lengths)
    world_lengths = torch.LongTensor(world_lengths)

    b = np.array([_pad_2d(x[1], max_target_len, b_pad=b_pad) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    c = np.array([_pad_2d(x[2], max_target_len, b_pad=b_pad) for x in batch],
                 dtype=np.float32)
    y_batch = torch.FloatTensor(c)


    rw = int(r * hparams.world_upsample)

    d = np.array([np.pad(x[3], (rw, max_world_len-len(x[3])-rw), mode = "constant") for x in batch],dtype=np.float32)
    f0_batch = torch.FloatTensor(d)
    e = np.array([_pad_2d(x[4], max_world_len, b_pad=rw) for x in batch],dtype=np.float32)
    sp_batch = torch.FloatTensor(e)
    f = np.array([_pad_2d(x[5], max_world_len, b_pad=rw) for x in batch],dtype=np.float32)
    ap_batch = torch.FloatTensor(f)


    # text positions
    text_positions = np.array([_pad(np.arange(1, len(x[0]) + 1), max_input_len)
                               for x in batch], dtype=np.int)
    text_positions = torch.LongTensor(text_positions)

    max_decoder_target_len = max_target_len // r

    # frame positions
    s, e = 1, max_decoder_target_len + 1
    # if b_pad > 0:
    #    s, e = s - 1, e - 1
    frame_positions = torch.arange(s, e).long().unsqueeze(0).expand(
        len(batch), max_decoder_target_len)

    # done flags
    done = np.array([_pad(np.zeros(len(x[1]) // r - 1),
                          max_decoder_target_len, constant_values=1)
                     for x in batch])
    done = torch.FloatTensor(done).unsqueeze(-1)

    #voiced flags
    voiced = np.array([np.pad(x[3]!=0, (rw, max_world_len-len(x[3])-rw), mode = "constant") for x in batch],dtype=np.float32)
    voiced_batch = torch.FloatTensor(voiced)

    if multi_speaker:
        speaker_ids = torch.LongTensor([x[6] for x in batch])
    else:
        speaker_ids = None

    return x_batch, input_lengths, mel_batch, y_batch, \
        (text_positions, frame_positions), done, target_lengths, speaker_ids, \
           (f0_batch, sp_batch, ap_batch, voiced_batch), world_lengths

if __name__== '__main__':
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path = args["--checkpoint"]
    checkpoint_seq2seq_path = args["--checkpoint-seq2seq"]
    checkpoint_postnet_path = args["--checkpoint-postnet"]
    load_embedding = args["--load-embedding"]
    checkpoint_restore_parts = args["--restore-parts"]
    speaker_id = args["--speaker-id"]
    speaker_id = int(speaker_id) if speaker_id is not None else None
    preset = args["--preset"]

    data_root = args["--data-root"]
    if data_root is None:
        data_root = join(dirname(__file__), "data", "ljspeech")

    log_event_path = args["--log-event-path"]
    reset_optimizer = args["--reset-optimizer"]

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())

    # Override hyper parameters
    hparams.parse(args["--hparams"])

    # Preventing Windows specific error such as MemoryError
    # Also reduces the occurrence of THAllocator.c 0x05 error in Widows build of PyTorch
    if platform.system() == "Windows":
        print(" [!] Windows Detected - IF THAllocator.c 0x05 error occurs SET num_workers to 1")

    assert hparams.name == "deepvoice3"
    print(hparams_debug_string())

    _frontend = getattr(frontend, hparams.frontend)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Input dataset definitions
    X = FileSourceDataset(TextDataSource(data_root, speaker_id))
    Mel = FileSourceDataset(MelSpecDataSource(data_root, speaker_id))
    Y = FileSourceDataset(LinearSpecDataSource(data_root, speaker_id))
    F0 = FileSourceDataset(F0DataSource(data_root, speaker_id))
    SP = FileSourceDataset(SpDataSource(data_root, speaker_id))
    AP = FileSourceDataset(ApDataSource(data_root, speaker_id))

    # Prepare sampler
    frame_lengths = Mel.file_data_source.frame_lengths
    sampler = PartialyRandomizedSimilarTimeLengthSampler(
        frame_lengths, batch_size=hparams.batch_size)

    # Dataset and Dataloader setup
    dataset = PyTorchDataset(X, Mel, Y, F0, SP, AP)
    data_loader = data_utils.DataLoader(
        dataset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers, sampler=sampler,
        collate_fn=collate_fn, pin_memory=hparams.pin_memory)

    device = torch.device("cuda" if use_cuda else "cpu")

    for x, input_lengths, mel, y, positions, done, target_lengths, speaker_ids, world, world_length in data_loader:
        start = time.time()
        ismultispeaker = speaker_ids is not None
        # Transform data to CUDA device
        text_positions, frame_positions = positions
        x = x.to(device)
        text_positions = text_positions.to(device)
        frame_positions = frame_positions.to(device)
        voiced, f0, sp, ap = world
        y = y.to(device)
        f0 = f0.to(device)
        sp = sp.to(device)
        ap = ap.to(device)
        voiced = voiced.to(device)
        mel, done = mel.to(device), done.to(device)
        target_lengths = target_lengths.to(device)
        speaker_ids = speaker_ids.to(device) if ismultispeaker else None

'''
