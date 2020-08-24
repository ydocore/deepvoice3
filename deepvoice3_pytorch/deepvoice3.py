# coding: utf-8

import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np

from .modules import Conv1d, ConvTranspose1d, Embedding, Linear, GradMultiply, Linear_relu
from .modules import get_mask_from_lengths, SinusoidalEncoding, Conv1dGLU


def expand_speaker_embed(inputs_btc, speaker_embed=None, tdim=1):
    if speaker_embed is None:
        return None
    # expand speaker embedding for all time steps
    # (B, N) -> (B, T, N)
    ss = speaker_embed.size()
    speaker_embed_btc = speaker_embed.unsqueeze(1).expand(
        ss[0], inputs_btc.size(tdim), ss[-1])
    return speaker_embed_btc


# エンコーダ
class Encoder(nn.Module):
    def __init__(self, n_vocab,
                 embed_dim,
                 n_speakers,
                 speaker_embed_dim,
                 padding_idx=None,
                 embedding_weight_std=0.1,
                 convolutions=((64, 5, .1),) * 7,
                 max_positions=512,
                 dropout=0.1,
                 apply_grad_scaling=True,
                 num_attention_layers=4):
        super(Encoder, self).__init__()
        self.dropout = dropout # ドロップアウト
        self.num_attention_layers = num_attention_layers # アテンション層の数(多分いらない)
        self.apply_grad_scaling = apply_grad_scaling # これなに？

        # テキストの埋め込み層
        # Text input embeddings
        self.embed_tokens = Embedding(
            n_vocab, embed_dim, padding_idx, embedding_weight_std)

        # 今は関係なし
        # Speaker embedding
        if n_speakers > 1:
            self.speaker_fc1 = Linear(speaker_embed_dim, embed_dim)
            self.speaker_fc2 = Linear(speaker_embed_dim, embed_dim)
        self.n_speakers = n_speakers

        # Non causual convolution blocks
        in_channels = embed_dim
        self.convolutions = nn.ModuleList()
        self.pre_linear = Linear(in_channels,convolutions[0][0]) # プレネット
        in_channels = convolutions[0][0]
        # 畳み込みブロック
        for idx, (out_channels, kernel_size, dilation) in enumerate(convolutions):
            self.convolutions.append(
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, out_channels, kernel_size, causal=False,
                          dilation=dilation, dropout=dropout, residual=True))
            in_channels = out_channels
        self.linear = Linear(in_channels,embed_dim) # ポストネット

    def forward(self, text_sequences, text_positions=None, lengths=None,
                speaker_embed=None):
        assert self.n_speakers == 1 or speaker_embed is not None

        # テキスト埋め込み
        # embed text_sequences
        x = self.embed_tokens(text_sequences.long())

        # 関係なし
        # expand speaker embedding for all time steps
        if speaker_embed is not None:
            x = x + F.softsign(self.speaker_fc1(speaker_embed))[:,None,:]

        input_embedding = x

        # プレネット
        # pre FC
        x = self.pre_linear(x)

        # データを変形
        # B x T x C (Batch x Time x Channel) -> B x C x T (Batch x Channel x Time)
        x = x.transpose(1, 2)

        # 畳み込みブロック
        # １D conv blocks
        for f in self.convolutions:
            x = f(x, speaker_embed) if isinstance(f, Conv1dGLU) else f(x)

        # データの変形を解除
        # Back to B x T x C (Batch x Time x Channel)
        x = x.transpose(1, 2)
        
        # ポストネット
        # post FC
        x = self.linear(x)
        keys = x

        # 関係なし
        if speaker_embed is not None:
            keys = keys + F.softsign(self.speaker_fc2(speaker_embed))[:,None,:]

        # よくわからん(勾配をスケーリングしている？)
        # scale gradients (this only affects backward, not forward)
        if self.apply_grad_scaling and self.num_attention_layers is not None:
            keys = GradMultiply.apply(keys, 1.0 / (self.num_attention_layers))

        # valueを求める
        # add output to input embedding for attention
        values = (keys + input_embedding) * math.sqrt(0.5)

        return keys, values


class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim, att_hid, n_speakers, speaker_embed_dim,
                 key_position_rate, query_position_rate, position_weight, max_positions=512,
                 dropout=0.1, window_ahead=3, window_backward=1, 
                 ):
        super(AttentionLayer, self).__init__()
        # Used for compute multiplier for positional encodings
        if n_speakers > 1:
            self.speaker_proj1 = Linear(speaker_embed_dim, 1)
            self.speaker_proj2 = Linear(speaker_embed_dim, 1)
            self.key_position_rate = nn.Parameter(torch.tensor(key_position_rate,requires_grad=True))
        else:
            self.speaker_proj1, self.speaker_proj2 = None, None
            self.key_position_rate = key_position_rate
        self.query_position_rate = query_position_rate
        self.position_weight = position_weight

        # Position encodings for query (decoder states) and keys (encoder states)
        self.position_enc = SinusoidalEncoding(
            max_positions, embed_dim)

        self.query_projection = Linear(conv_channels, att_hid)
        self.key_projection = Linear(embed_dim, att_hid)
        # According to the DeepVoice3 paper, intiailize weights to same values
        self.key_projection.load_state_dict(self.query_projection.state_dict())
        self.value_projection = Linear(embed_dim, att_hid)

        self.out_projection = Linear(att_hid, conv_channels)
        self.dropout = dropout
        self.window_ahead = window_ahead
        self.window_backward = window_backward

    def forward(self, query, encoder_out, text_positions=None, frame_positions=None,
                speaker_embed=None, key_enc=None, query_enc=None,
                incremental=False, mask=None, last_attended=None):
        keys, values = encoder_out

        # position encodings
        if text_positions is not None:
            # TODO: may be useful to have projection per attention layer
            if self.speaker_proj1 is not None:
                w = self.key_position_rate * torch.sigmoid(self.speaker_proj1(speaker_embed)).view(-1)
            else:
                w = self.key_position_rate
            text_pos_enc = self.position_weight * self.position_enc(text_positions, w)
            keys = keys + text_pos_enc[:,:keys.size(1), :]
        if frame_positions is not None:
            if self.speaker_proj2 is not None:
                w = 2 * torch.sigmoid(self.speaker_proj2(speaker_embed)).view(-1)
            else:
                w = self.query_position_rate
            frame_pos_enc = self.position_weight * self.position_enc(frame_positions, w)
            query = query + frame_pos_enc[:,:frame_positions.size(-1),:] if not incremental else query + frame_pos_enc[:, frame_positions[0], :]

        keys = keys.contiguous()

        if self.value_projection is not None:
           values = self.value_projection(values)
        # TODO: yes, this is inefficient
        if self.key_projection is not None:
            keys = self.key_projection(keys)

        # attention
        x = self.query_projection(query)
        x = torch.bmm(x, keys.transpose(1,2))

        mask_value = -float("inf")
        if mask is not None:
            mask = mask.view(query.size(0), 1, -1)
            x.data.masked_fill_(mask, mask_value)

        if last_attended is not None:
            backward = last_attended - self.window_backward
            if backward > 0:
                x[:, :, :backward] = mask_value
            ahead = last_attended + self.window_ahead
            if ahead < x.size(-1):
                x[:, :, ahead:] = mask_value

        # softmax over last dim
        # (B, tgt_len, src_len)
        #sz = x.size()
        x = F.softmax(x, dim=2)#.view(sz[0] * sz[1], sz[2]), dim=1)
        #x = x.view(sz)
        attn_scores = x

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = torch.bmm(x, values)

        # scale attention output
        s = values.size(1)
        x = x *(math.sqrt(1.0/s))

        # project back
        x = self.out_projection(x)
        return x, attn_scores


# デコーダ
class Decoder(nn.Module):
    def __init__(self,
                 embed_dim,
                 attention_hidden,
                 n_speakers,
                 speaker_embed_dim,
                 in_dim=80,
                 r=5,
                 max_positions=512,
                 padding_idx=None,
                 preattention=(80,128),
                 convolutions=((128, 5, 1),) * 4,
                 dropout=0.1,
                 use_memory_mask=False,
                 force_monotonic_attention=False,
                 query_position_rate=1.0,
                 key_position_rate=1.29,
                 position_weight=1.0,
                 window_ahead=3,
                 window_backward=1,
                 ):
        super(Decoder, self).__init__()
        self.dropout = dropout # ドロップアウト
        self.in_dim = in_dim # 入力次元
        self.r = r # 出力フレーム数
        self.query_position_rate = query_position_rate # 位置率(クエリ)
        self.key_position_rate = torch.tensor(key_position_rate) if n_speakers > 1 else key_position_rate # 位置率(キー)
        self.position_weight = position_weight # わからん

        # 入力チャンネル数
        in_channels = in_dim*r

        # プレネット
        # Prenet: FC layer
        self.preattention = nn.ModuleList()
        self.speaker_fc = nn.ModuleList()
        for _, out_channels in preattention:
            if n_speakers > 1:
                self.speaker_fc.append(Linear(speaker_embed_dim,in_channels)) #apply multi-speaker
            else:
                self.speaker_fc.append(None)
            self.preattention.append(Linear(in_channels,out_channels))
            in_channels = out_channels

        # 因果的畳み込み＋アテンション
        # Causal convolution blocks + attention layers
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()
        for i, (out_channels, kernel_size, dilation) in enumerate(convolutions):
            assert in_channels == out_channels
            self.convolutions.append(
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, out_channels, kernel_size, causal=True,
                          dilation=dilation, dropout=dropout, residual=True))
            self.attention.append(
                AttentionLayer(out_channels, embed_dim,attention_hidden,
                               dropout=dropout,
                               window_ahead=window_ahead,
                               window_backward=window_backward,
                               n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
                               key_position_rate=key_position_rate, query_position_rate=query_position_rate,
                               position_weight=position_weight))
            in_channels = out_channels

        # ポストネット(mel)
        self.last_fc = Linear(in_channels,in_dim*r)
        self.gate_fc = Linear(in_channels,in_dim*r)

        # ポストネット(done)
        # Mel-spectrogram (before sigmoid) -> Done binary flag
        self.fc = Linear(in_channels, 1)

        self.max_decoder_steps = 200
        self.min_decoder_steps = 10
        self.use_memory_mask = use_memory_mask
        # 単調強制を行うか
        if isinstance(force_monotonic_attention, bool):
            self.force_monotonic_attention = [force_monotonic_attention] * len(convolutions)
        else:
            self.force_monotonic_attention = force_monotonic_attention

    # 入力はなに？(正解のメルなの？)
    def forward(self,
                encoder_out,
                inputs=None,
                text_positions=None,
                frame_positions=None,
                speaker_embed=None,
                lengths=None):
        # 初回の場合
        if inputs is None:
            assert text_positions is not None
            self.start_fresh_sequence() # プレネットと畳み込みをクリア？
            outputs = self.incremental_forward(encoder_out, text_positions, speaker_embed)
            return outputs

        # 多分フレーム数に合わせて入力を変形している？(入力サイズの確認が必要)
        # Grouping multiple frames if necessary
        if inputs.size(-1) == self.in_dim:
            inputs = inputs.view(inputs.size(0), inputs.size(1) // self.r, -1)
        assert inputs.size(-1) == self.in_dim * self.r

        # keyとvalueの取得
        keys, values = encoder_out

        # 多分今のところは関係なし
        if self.use_memory_mask and lengths is not None:
            mask = get_mask_from_lengths(keys, lengths)
        else:
            mask = None

        # transpose only once to speed up attention layers
        #keys = keys.contiguous()

        x = inputs

        # プレネット
        # Prenet
        for i, (f, sf) in enumerate(zip(self.preattention, self.speaker_fc)):
            # 2層目以降はドロップアウト
            if i > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            if speaker_embed is not None:
                x = x + F.softsign(sf(speaker_embed))[:,None,:]
            x = F.relu(f(x))

        # xのサイズを変換
        # Generic case: B x T x C -> B x C x T
        x = x.transpose(1, 2)

        # 畳み込み+アテンション
        # Casual convolutions + Multi-hop attentions
        alignments = []
        for i, (f, attention) in enumerate(zip(self.convolutions, self.attention)):

            # 畳み込み(なんで場合わけをしてるのかわかんないけど多分そんな関係なさそう)
            x = f(x, speaker_embed) if isinstance(f, Conv1dGLU) else f(x)
            residual = x

            # アテンション
            # Feed conv output to attention layer as query
            if attention is not None:
                assert isinstance(f, Conv1dGLU)
                # (B x T x C)
                x = x.transpose(1, 2)
                x, alignment = attention(x, (keys, values), text_positions=text_positions,
                                         frame_positions=frame_positions, speaker_embed=speaker_embed)
                # (T x B x C)
                x = x.transpose(1, 2)
                alignments += [alignment]

            # 残差接続とスケーリング(if文の意味がわからない)
            if isinstance(f, Conv1dGLU):
                x = (x + residual) * math.sqrt(0.5)

        # 畳み込み＋アテンションの出力をdecoder stateとして取得
        # decoder state (B x T x C)
        # internal representation before compressed to output dimention
        decoder_states = x.transpose(1, 2).contiguous()

        # サイズを変換
        # Back to B x T x C
        x = x.transpose(1, 2)
        out = self.last_fc(x) # ポストネット
        gate = self.gate_fc(x) # ポストネット(gate)

        # project to mel-spectorgram
        outputs = torch.sigmoid(gate) * out # [疑問] なぜシグモイドを通したgateをかけている？
        outputs = outputs.view(outputs.size(0),-1,self.in_dim) # サイズを変換

        # doneを求める
        # Done flag
        done = torch.sigmoid(self.fc(x))

        return outputs, torch.stack(alignments), done, decoder_states

    # 初回の時に使用
    def incremental_forward(self,
                            encoder_out,
                            text_positions,
                            speaker_embed=None,
                            initial_input=None,
                            test_inputs=None):
        # エンコーダの出力を取得
        keys, values = encoder_out
        B = keys.size(0)

        # transpose only once to speed up attention layers
        #keys = keys.transpose(1, 2).contiguous()

        decoder_states = []
        outputs = []
        alignments = []
        dones = []
        # 単調強制に関する何か？
        # intially set to zeros
        last_attended = [None] * len(self.attention)
        for idx, v in enumerate(self.force_monotonic_attention):
            last_attended[idx] = 0 if v else None

        num_attention_layers = sum([layer is not None for layer in self.attention])
        t = 0
        # 最初の入力を設定？
        if initial_input is None:
            initial_input = keys.data.new(B, 1, self.in_dim *self.r).zero_()
        current_input = initial_input
        
        while True:
            # なんやこれ
            # frame pos start with 1.
            frame_pos = keys.data.new(B, 1).fill_(t + 1).long()

            # 何かをテストする場合の入力を設定？
            if test_inputs is not None:
                if t >= test_inputs.size(1):
                    break
                current_input = test_inputs[:, t, :].unsqueeze(1)
            else:
                if t > 0:
                    current_input = outputs[-1]
            x = current_input

            # プレネット
            # Prenet
            for f, sf in zip(self.preattention, self.speaker_fc):
                x = F.dropout(x, p=self.dropout, training=self.training)
                if speaker_embed is not None:
                    x = x + F.softsign(sf(speaker_embed))[:,None,:]
                x = F.relu(f(x))

            # 畳み込みブロック＋アテンションブロック
            # Casual convolutions + Multi-hop attentions
            ave_alignment = None
            for idx, (f, attention) in enumerate(zip(self.convolutions,
                                                     self.attention)):
                # 畳み込みブロック
                if isinstance(f, Conv1dGLU):
                    x = f.incremental_forward(x, speaker_embed)
                else:
                    try:
                        x = f.incremental_forward(x)
                    except AttributeError as e:
                        x = f(x)

                residual = x

                # アテンションブロック
                # attention
                if attention is not None:
                    assert isinstance(f, Conv1dGLU)
                    x, alignment = attention(x, (keys, values), text_positions=text_positions,
                             frame_positions=frame_pos, speaker_embed=speaker_embed,
                             incremental=True, last_attended=last_attended[idx])
                    # 単調強制を実行
                    if self.force_monotonic_attention[idx]:
                        last_attended[idx] = alignment.max(-1)[1].view(-1).data[0]

                    # なにこれ？
                    if ave_alignment is None:
                        #ave_alignment = alignment
                        ave_alignment = torch.zeros([len(self.attention),alignment.size(0),alignment.size(1),alignment.size(-1)])
                    #else:
                    #    ave_alignment = ave_alignment + ave_alignment

                    ave_alignment[idx] = alignment
                # 残左接続
                # residual
                if isinstance(f, Conv1dGLU):
                    x = (x + residual) * math.sqrt(0.5)

            decoder_state = x
            out = self.last_fc(x) # ポストネット
            gate = self.gate_fc(x) # ポストネット(gate)

            #output & done flag predictions
            output = torch.sigmoid(gate) * out # gateを掛け合わせた値を求める
            done = torch.sigmoid(self.fc(x)) # 最終フレーム予測を求める

            decoder_states += [decoder_state]
            outputs += [output]
            alignments += [ave_alignment]
            dones += [done]


            # [確認] .all()の中身
            # .all()ってなに？
            t += 1
            if test_inputs is None:
                if (done > 0.5).all() and t > self.min_decoder_steps:
                    break
                elif t > self.max_decoder_steps:
                    break

        # サイズが1の次元を削除
        # Remove 1-element time axis
        #for idx, alignment in enumerate(alignments):
        alignments = list(map(lambda x: x.squeeze(1), alignments))
        decoder_states = list(map(lambda x: x.squeeze(1), decoder_states))
        outputs = list(map(lambda x: x.squeeze(1), outputs))

        # 配列の中身を結合
        # Combine outputs for all time steps
        alignments = torch.stack(alignments).transpose(0, 2)
        decoder_states = torch.stack(decoder_states).transpose(0, 1).contiguous()
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()
        outputs = outputs.view(outputs.size(0),-1,self.in_dim)


        return outputs, alignments, dones, decoder_states

    # プレネットと畳み込みブロックのバッファをクリアする
    def start_fresh_sequence(self):
        _clear_modules(self.preattention)
        _clear_modules(self.convolutions)


# バッファをクリアする
def _clear_modules(modules):
    for m in modules:
        try:
            m.clear_buffer()
        except AttributeError as e:
            pass


class Converter(nn.Module):
    def __init__(self, n_speakers, speaker_embed_dim,
                 in_dim, convolutions=((256, 5, 1),) * 4, r=5, dropout=0.1):
        super(Converter, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.n_speakers = n_speakers
        self.r = r

        # Non causual convolution blocks
        in_channels = convolutions[0][0]
        # Idea from nyanko
        self.convolutions = nn.ModuleList()
        for i, (out_channels, kernel_size, dilation) in enumerate(convolutions):
            self.convolutions.append(
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, out_channels, kernel_size, causal=False,
                          dilation=dilation, dropout=dropout, residual=True))
            in_channels = out_channels
        # Last fully connect
        self.fc = Linear(in_channels,in_channels*self.r)


    def forward(self, x, speaker_embed=None):
        assert self.n_speakers == 1 or speaker_embed is not None
        # Generic case: B x T x C -> B x C x T
        x = x.transpose(1, 2)

        for f in self.convolutions:
            x = f(x, speaker_embed) if isinstance(f, Conv1dGLU) else f(x)

        # Back to B x T x C
        x = x.transpose(1, 2)
        x = self.fc(x)

        return x.view(x.size(0), -1, x.size(-1)//self.r)

class LinearConverter(nn.Module):
    def __init__(self, n_speakers, speaker_embed_dim,
                 in_dim, out_dim, convolutions=((256, 5, 1),) * 4, r=5, dropout=0.1):
        super(LinearConverter, self).__init__()
        self.conv_block = Converter(n_speakers, speaker_embed_dim, in_dim, convolutions, r, dropout)
        in_channels = convolutions[0][0]
        self.linear_spec = Linear(in_channels, out_dim)

    def forward(self, x, speaker_embed=None):
        x = self.conv_block(x, speaker_embed)
        x = self.linear_spec(x)

        return x

class WorldConverter(nn.Module):
    def __init__(self, n_speakers, speaker_embed_dim,
                 in_dim, out_dim, convolutions=((256, 5, 1),) * 4,
                 time_upsampling=1, r=5, dropout=0.1):
        super(WorldConverter, self).__init__()
        self.conv_block = Converter(n_speakers, speaker_embed_dim, in_dim, convolutions, r, dropout)
        in_channels = convolutions[0][0]

        # world parameter
        self.upsample = nn.Upsample(scale_factor=time_upsampling)
        self.world = Conv1dGLU(n_speakers, speaker_embed_dim,
                               in_channels, in_channels, kernel_size=5, causal=False,
                               dilation=1, dropout=dropout, residual=True)
        self.voiced = Linear(in_channels, 1)
        self.f0 = Linear(in_channels, 1)
        self.sp = Linear(in_channels, out_dim)
        self.ap = Linear(in_channels, out_dim)

    def forward(self, x, speaker_embed=None):
        x = self.conv_block(x, speaker_embed)
        x = self.upsample(x.transpose(1,2))
        x = self.world(x, speaker_embed)
        x = x.transpose(1,2)

        voiced = torch.sigmoid(self.voiced(x))
        f0 = self.f0(x)
        sp = self.sp(x)
        ap = self.ap(x)

        return (voiced[:,:,0], f0[:,:,0], sp, ap)
