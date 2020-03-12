import hparam_tf.hparam

# NOTE: If you want full control for model architecture. please take a look
# at the code and change whatever you want. Some hyper parameters are hardcoded.

# Default hyperparameters:
hparams = hparam_tf.hparam.HParams(
    name="deepvoice3",

    # Text:
    # [en, jp]
    frontend='en',

    # Replace words to its pronunciation with fixed probability.
    # e.g., 'hello' to 'HH AH0 L OW1'
    # [en, jp]
    # en: Word -> pronunciation using CMUDict
    # jp: Word -> pronounciation usnig MeCab
    # [0 ~ 1.0]: 0 means no replacement happens.
    replace_pronunciation_prob=0.5,

    # Convenient model builder
    # [deepvoice3, deepvoice3_multispeaker, nyanko]
    # Definitions can be found at deepvoice3_pytorch/builder.py
    # deepvoice3: DeepVoice3 https://arxiv.org/abs/1710.07654
    # deepvoice3_multispeaker: Multi-speaker version of DeepVoice3
    # nyanko: https://arxiv.org/abs/1710.08969
    builder="deepvoice3",

    # Must be configured depends on the dataset and model you use
    n_speakers=108,
    speaker_embed_dim=16,

    # Audio:
    num_mels=80,
    fmin=125,
    fmax=7600,
    fft_size=4096,
    fft_wsize=2400,
    hop_size=600, #fft_wsize/4
    sample_rate=48000,
    preemphasis=0.97,
    min_level_db=-100,
    spec_ref_level_db=20, #max_db : 40
    sp_ref_level_db=20, #max_db : 20
    f0_norm=400,
    #WORLDのフレームサイズはSpectrogramのフレームサイズの定数倍で求めることが出来ないので，
    #WORLDのフレームサイズ/Spectrogramのフレームサイズ が最大の値でupsampleする
    # can be computed by `compute_timestamp_ratio.py`.
    world_upsample=2.6,
    sp_fft_size=1025, #can compute pyworld.get_cheaptrick_fft_size(fs) //2 + 1
    # whether to rescale waveform or not.
    # Let x is an input waveform, rescaled waveform y is given by:
    # y = x / np.abs(x).max() * rescaling_max
    rescaling=False,
    rescaling_max=0.999,
    # mel-spectrogram is normalized to [0, 1] for each utterance and clipping may
    # happen depends on min_level_db and ref_level_db, causing clipping noise.
    # If False, assertion is added to ensure no clipping happens.
    allow_clipping_in_normalization=True,

    # Model:
    outputs_per_step=4,  # must be 1 when builder="nyanko"
    embedding_weight_std=1,
    speaker_embedding_weight_std=0.01,
    padding_idx=0,
    # Maximum number of input text length
    # try setting larger value if you want to give very long text input
    max_positions=2048,
    dropout=1-0.95,
    kernel_size=5,
    text_embed_dim=256,
    encoder_channels=128,
    num_encoder_layer=7,
    decoder_channels=256,
    num_decoder_layer=6,
    attention_hidden=256,
    # Note: large converter channels requires significant computational cost
    converter_channels=256,
    num_converter_layer=6,
    query_position_rate=1.0,
    # can be computed by `compute_timestamp_ratio.py`.
    key_position_rate=1.8,  # 2.37 for jsut
    position_weight=0.1,
    use_memory_mask=False,
    trainable_positional_encodings=True,
    freeze_embedding=False,
    # If True, use decoder's internal representation for postnet inputs,
    # otherwise use mel-spectrogram.
    use_decoder_state_for_postnet_input=True,

    # Data loader
    pin_memory=False,
    num_workers=5,  # Set it to 1 when in Windows (MemoryError, THAllocator.c 0x5)

    # Training:
    batch_size=16,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_eps=1e-8,
    amsgrad=False,
    initial_learning_rate=5e-4,
    lr_schedule="step_learning_rate_decay",
    lr_schedule_kwargs={},
    nepochs=1001,
    weight_decay=0.0,
    max_clip=100.0,
    clip_thresh=5.0,

    # Save
    checkpoint_interval=5000,  #test
    eval_interval=50000,
    save_optimizer_state=True,

    # Eval:
    # this can be list for multple layers of attention
    # e.g., [True, False, False, False, True]
    force_monotonic_attention=[False,False,False,False,False,False],
    # Attention constraint for incremental decoding
    window_ahead=3,
    # 0 tends to prevent word repretetion, but sometime causes skip words
    window_backward=0,
    power=1.4,  # Power to raise magnitudes to prior to phase retrieval

    # GC:
    # Forced garbage collection probability
    # Use only when MemoryError continues in Windows (Disabled by default)
    #gc_probability = 0.001,

    # json_meta mode only
    # 0: "use all",
    # 1: "ignore only unmatched_alignment",
    # 2: "fully ignore recognition",
    ignore_recognition_level=2,
    # when dealing with non-dedicated speech dataset(e.g. movie excerpts), setting min_text above 15 is desirable. Can be adjusted by dataset.
    min_text=20,
    # if true, data without phoneme alignment file(.lab) will be ignored
    process_only_htk_aligned=False,
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
#TODO:不要なパラメータ削除