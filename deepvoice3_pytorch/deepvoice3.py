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


class Encoder(nn.Module):
    def __init__(self, n_vocab, embed_dim, n_speakers, speaker_embed_dim,
                 padding_idx=None, embedding_weight_std=0.1,
                 convolutions=((64, 5, .1),) * 7,
                 max_positions=512, dropout=0.1, apply_grad_scaling=True, num_attention_layers=4):
        super(Encoder, self).__init__()
        self.dropout = dropout
        self.num_attention_layers = num_attention_layers
        self.apply_grad_scaling = apply_grad_scaling

        # Text input embeddings
        self.embed_tokens = Embedding(
            n_vocab, embed_dim, padding_idx, embedding_weight_std)

        # Speaker embedding
        if n_speakers > 1:
            self.speaker_fc1 = Linear(speaker_embed_dim, embed_dim, dropout=dropout)
            self.speaker_fc2 = Linear(speaker_embed_dim, embed_dim, dropout=dropout)
        self.n_speakers = n_speakers

        # Non causual convolution blocks
        in_channels = embed_dim
        self.convolutions = nn.ModuleList()
        std_mul = 4.0
        self.pre_linear = Linear(in_channels,convolutions[0][0])
        in_channels = convolutions[0][0]
        for idx, (out_channels, kernel_size, dilation) in enumerate(convolutions):
            if idx+1 == len(convolutions):
                std_mul= 1.0
            self.convolutions.append(
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, out_channels, kernel_size, causal=False,
                          dilation=dilation, dropout=dropout, std_mul=std_mul,
                          residual=True))
            in_channels = out_channels
        self.linear = Linear(in_channels,embed_dim)#original

    def forward(self, text_sequences, text_positions=None, lengths=None,
                speaker_embed=None):
        assert self.n_speakers == 1 or speaker_embed is not None

        # embed text_sequences
        x = self.embed_tokens(text_sequences.long())
        #x = F.dropout(x, p=self.dropout, training=self.training)

        # expand speaker embedding for all time steps
        speaker_embed_btc = expand_speaker_embed(x, speaker_embed)
        if speaker_embed_btc is not None:
            #speaker_embed_btc = F.dropout(speaker_embed_btc, p=self.dropout, training=self.training)
            x = x + F.softsign(self.speaker_fc1(speaker_embed_btc))

        input_embedding = x

        x = self.pre_linear(x)

        # B x T x C -> B x C x T
        x = x.transpose(1, 2)

        # １D conv blocks
        for f in self.convolutions:
            x = f(x, speaker_embed_btc) if isinstance(f, Conv1dGLU) else f(x)

        # Back to B x T x C
        x = x.transpose(1, 2)
        x = self.linear(x)
        keys = x

        if speaker_embed_btc is not None:
            keys = keys + F.softsign(self.speaker_fc2(speaker_embed_btc))

        # scale gradients (this only affects backward, not forward)
        if self.apply_grad_scaling and self.num_attention_layers is not None:
            keys = GradMultiply.apply(keys, 1.0 / (self.num_attention_layers))

        # add output to input embedding for attention
        values = (keys + input_embedding) * math.sqrt(0.5)

        return keys, values


class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim, att_hid,dropout=0.1,
                 window_ahead=3, window_backward=1,
                 key_projection=True, value_projection=True):
        super(AttentionLayer, self).__init__()
        self.query_projection = Linear(conv_channels, att_hid)
        if key_projection:
            self.key_projection = Linear(embed_dim, att_hid)
            # According to the DeepVoice3 paper, intiailize weights to same values
            # TODO: Does this really work well? not sure..
            if conv_channels == embed_dim:
                self.key_projection.load_state_dict(self.query_projection.state_dict())
        else:
            self.key_projection = None
        if value_projection:
            self.value_projection = Linear(embed_dim, att_hid)
        else:
            self.value_projection = None

        self.out_projection = Linear(att_hid, conv_channels)
        self.dropout = dropout
        self.window_ahead = window_ahead
        self.window_backward = window_backward

    def forward(self, query, encoder_out, mask=None, last_attended=None):
        keys, values = encoder_out
        residual = query
        if self.value_projection is not None:
           values = self.value_projection(values)
        # TODO: yes, this is inefficient
        if self.key_projection is not None:
            keys = self.key_projection(keys.transpose(1, 2)).transpose(1, 2)

        # attention
        x = self.query_projection(query)
        x = torch.bmm(x, keys)

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
        x = x *(s * math.sqrt(1.0/s))

        # project back
        x = self.out_projection(x)
        #x = (x + residual) * math.sqrt(0.5)
        return x, attn_scores


class Decoder(nn.Module):
    def __init__(self, embed_dim, attention_hidden, n_speakers, speaker_embed_dim,
                 in_dim=80, r=5,
                 max_positions=512, padding_idx=None,
                 preattention=(80,128),
                 convolutions=((128, 5, 1),) * 4,
                 attention=True, dropout=0.1,
                 use_memory_mask=False,
                 force_monotonic_attention=False,
                 query_position_rate=1.0,
                 key_position_rate=1.29,
                 position_weight=1.0,
                 window_ahead=3,
                 window_backward=1,
                 key_projection=True,
                 value_projection=True,
                 ):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.att_hid = attention_hidden
        self.r = r
        self.query_position_rate = query_position_rate
        self.key_position_rate = key_position_rate
        self.position_weight = position_weight

        in_channels = in_dim*r
        if isinstance(attention, bool):
            # expand True into [True, True, ...] and do the same with False
            attention = [attention] * len(convolutions)
        self.att = attention

        # Position encodings for query (decoder states) and keys (encoder states)
        self.embed_query_positions = SinusoidalEncoding(
            max_positions, convolutions[0][0])
        self.embed_keys_positions = SinusoidalEncoding(
            max_positions, embed_dim)
        # Used for compute multiplier for positional encodings
        if n_speakers > 1:
            self.speaker_proj1 = Linear(speaker_embed_dim, in_dim, dropout=dropout)
            self.speaker_proj2 = Linear(speaker_embed_dim, in_dim, dropout=dropout)
        else:
            self.speaker_proj1, self.speaker_proj2 = None, None

        # Prenet: causal convolution blocks
        self.preattention = nn.ModuleList()
        '''元論文に変更
        in_channels = in_dim * r

        std_mul = 4.0

        for out_channels, kernel_size, dilation in preattention:
            if in_channels != out_channels:
                # Conv1d + ReLU
                self.preattention.append(
                    Conv1d(in_channels, out_channels, kernel_size=1, padding=0,
                           dilation=1, std_mul=std_mul))
                self.preattention.append(nn.ReLU(inplace=True))
                in_channels = out_channels
                std_mul = 2.0
            self.preattention.append(
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, out_channels, kernel_size, causal=True,
                          dilation=dilation, dropout=dropout, std_mul=std_mul,
                          residual=True))

            in_channels = out_channels
            std_mul = 4.0
            '''
        in_channels = in_dim*r
        for _, out_channels in preattention:
            self.preattention.append(Linear_relu(in_channels,out_channels))
            self.preattention.append(nn.ReLU(inplace=True))
            in_channels = out_channels



        in_channels = out_channels
        std_mul=4.0
        # Causal convolution blocks + attention layers
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()

        for i, (out_channels, kernel_size, dilation) in enumerate(convolutions):
            assert in_channels == out_channels
            if i+1 == len(convolutions):
                std_mul=1.0
            self.convolutions.append(
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, out_channels, kernel_size, causal=True,
                          dilation=dilation, dropout=dropout, std_mul=std_mul,
                          residual=True))
            self.attention.append(
                AttentionLayer(out_channels, embed_dim,attention_hidden,
                               dropout=dropout,
                               window_ahead=window_ahead,
                               window_backward=window_backward,
                               key_projection=key_projection,
                               value_projection=value_projection))
            in_channels = out_channels
            #std_mul = 4.0
        # Last 1x1 convolution
        self.last_conv = Conv1d(in_channels, in_dim * r, kernel_size=1,
                                padding=0, dilation=1, std_mul=std_mul,
                                dropout=dropout)

        self.last_fc = Linear(in_channels,in_dim*r)
        self.gate_fc = Linear(in_channels,in_dim*r)

        # Mel-spectrogram (before sigmoid) -> Done binary flag
        self.fc = Linear(in_channels, 1)

        self.max_decoder_steps = 200
        self.min_decoder_steps = 10
        self.use_memory_mask = use_memory_mask
        if isinstance(force_monotonic_attention, bool):
            self.force_monotonic_attention = [force_monotonic_attention] * len(convolutions)
        else:
            self.force_monotonic_attention = force_monotonic_attention

    def forward(self, encoder_out, inputs=None,
                text_positions=None, frame_positions=None,
                speaker_embed=None, lengths=None):
        if inputs is None:
            assert text_positions is not None
            self.start_fresh_sequence()
            outputs = self.incremental_forward(encoder_out, text_positions, speaker_embed)
            return outputs

        # Grouping multiple frames if necessary
        if inputs.size(-1) == self.in_dim:
            inputs = inputs.view(inputs.size(0), inputs.size(1) // self.r, -1)
        assert inputs.size(-1) == self.in_dim * self.r

        keys, values = encoder_out

        if self.use_memory_mask and lengths is not None:
            mask = get_mask_from_lengths(keys, lengths)
        else:
            mask = None

        # position encodings
        if text_positions is not None:
            #w = 1 #position weight
            # TODO: may be useful to have projection per attention layer
            if self.speaker_proj1 is not None:
                w = self.key_position_rate * torch.sigmoid(self.speaker_proj1(speaker_embed)).view(-1)
            else:
                w = self.key_position_rate
            keys = keys + self.embed_keys_positions(text_positions, w)[:,:text_positions.size(-1),:]
        if frame_positions is not None:
            #w = 1 #self.query_position_rate
            if self.speaker_proj2 is not None:
                w = 2 * torch.sigmoid(self.speaker_proj2(speaker_embed)).view(-1)
            else:
                w = self.query_position_rate
            frame_pos_embed = self.embed_query_positions(frame_positions, w)

        # transpose only once to speed up attention layers
        keys = keys.transpose(1, 2).contiguous()

        x = inputs

        # expand speaker embedding for all time steps
        speaker_embed_btc = expand_speaker_embed(inputs, speaker_embed)


        # Generic case: B x T x C -> B x C x T
        #x = x.transpose(1, 2)

        # Prenet
        for i, f in enumerate(self.preattention):
            if i > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            if speaker_embed_btc is not None:
                #speaker_embed_btc = F.dropout(speaker_embed_btc, p=self.dropout, training=self.training)
                x = x + F.softsign(self.speaker_fc1(speaker_embed_btc))
            x = f(x, speaker_embed_btc) if isinstance(f, Conv1dGLU) else f(x)

        # Generic case: B x T x C -> B x C x T
        x = x.transpose(1, 2)

        # Casual convolutions + Multi-hop attentions
        alignments = []
        for i, (f, attention) in enumerate(zip(self.convolutions, self.attention)):

            x = f(x, speaker_embed_btc) if isinstance(f, Conv1dGLU) else f(x)
            residual = x

            # Feed conv output to attention layer as query
            if attention is not None:
                assert isinstance(f, Conv1dGLU)
                # (B x T x C)
                x = x.transpose(1, 2)
                x = x if frame_positions is None else x + frame_pos_embed[:,:frame_positions.size(-1),:]
                x, alignment = attention(x, (keys, values), mask=mask)
                # (T x B x C)
                x = x.transpose(1, 2)
                alignments += [alignment]


            if isinstance(f, Conv1dGLU):
                x = (x + residual) * math.sqrt(0.5)

        # decoder state (B x T x C):
        # internal representation before compressed to output dimention
        decoder_states = x.transpose(1, 2).contiguous()
        #x = self.last_conv(x) linearでxをメルスぺの次元数にしなきゃ

        # Back to B x T x C
        x = x.transpose(1, 2)
        out = self.last_fc(x)
        gate = self.gate_fc(x)

        # project to mel-spectorgram
        outputs = torch.sigmoid(gate) * out
        outputs = outputs.view(outputs.size(0),-1,self.in_dim)

        # Done flag
        done = torch.sigmoid(self.fc(x))



        return outputs, torch.stack(alignments), done, decoder_states

    def incremental_forward(self, encoder_out, text_positions, speaker_embed=None,
                            initial_input=None, test_inputs=None):
        keys, values = encoder_out
        B = keys.size(0)

        # position encodings
        w = self.key_position_rate
        # TODO: may be useful to have projection per attention layer
        if self.speaker_proj1 is not None:
            w = w * torch.sigmoid(self.speaker_proj1(speaker_embed)).view(-1)
        text_pos_embed = self.embed_keys_positions(text_positions, w)
        keys = keys + text_pos_embed[:,:text_positions.size(-1),:]

        # transpose only once to speed up attention layers
        keys = keys.transpose(1, 2).contiguous()

        decoder_states = []
        outputs = []
        alignments = []
        dones = []
        # intially set to zeros
        last_attended = [None] * len(self.attention)
        for idx, v in enumerate(self.force_monotonic_attention):
            last_attended[idx] = 0 if v else None

        num_attention_layers = sum([layer is not None for layer in self.attention])
        t = 0
        if initial_input is None:
            initial_input = keys.data.new(B, 1, self.in_dim *self.r).zero_()#in_dim*r del
        current_input = initial_input
        while True:
            # frame pos start with 1.
            frame_pos = keys.data.new(B, 1).fill_(t + 1).long()
            w = self.query_position_rate
            if self.speaker_proj2 is not None:
                w = w * torch.sigmoid(self.speaker_proj2(speaker_embed)).view(-1)
            frame_pos_embed = self.embed_query_positions(frame_pos, w)

            if test_inputs is not None:
                if t >= test_inputs.size(1):
                    break
                current_input = test_inputs[:, t, :].unsqueeze(1)
            else:
                if t > 0:
                    current_input = outputs[-1]
            x = current_input

            # expand speaker embedding for all time steps
            speaker_embed_btc = expand_speaker_embed(initial_input, speaker_embed)

            # Prenet
            for f in self.preattention:
                x = F.dropout(x, p=self.dropout, training=self.training)
                if speaker_embed_btc is not None:
                    speaker_embed_btc = F.dropout(speaker_embed_btc, p=self.dropout, training=self.training)
                    x = x + F.softsign(self.speaker_fc1(speaker_embed_btc))
                x=f(x)
                '''
                if isinstance(f, Conv1dGLU):
                    x = f.incremental_forward(x, speaker_embed)
                else:
                    try:
                        x = f.incremental_forward(x)
                    except AttributeError as e:
                        x = f(x)
                '''

            # Casual convolutions + Multi-hop attentions
            ave_alignment = None
            for idx, (f, attention) in enumerate(zip(self.convolutions,
                                                     self.attention)):
                if isinstance(f, Conv1dGLU):
                    x = f.incremental_forward(x, speaker_embed)
                else:
                    try:
                        x = f.incremental_forward(x)
                    except AttributeError as e:
                        x = f(x)

                residual = x

                # attention
                if attention is not None:
                    assert isinstance(f, Conv1dGLU)
                    x = x + frame_pos_embed[:,frame_pos.size(-1),:]
                    x, alignment = attention(x, (keys, values),
                                             last_attended=last_attended[idx])
                    if self.force_monotonic_attention[idx]:
                        last_attended[idx] = alignment.max(-1)[1].view(-1).data[0]

                    if ave_alignment is None:
                        #ave_alignment = alignment
                        ave_alignment = torch.zeros([len(self.attention),alignment.size(0),alignment.size(1),alignment.size(-1)])
                    #else:
                    #    ave_alignment = ave_alignment + ave_alignment

                    ave_alignment[idx] = alignment
                # residual
                if isinstance(f, Conv1dGLU):
                    x = (x + residual) * math.sqrt(0.5)

            decoder_state = x
            out = self.last_fc(x)
            #ave_alignment = ave_alignment.div_(num_attention_layers)

            # Ooutput & done flag predictions
            output = torch.sigmoid(out)
            done = torch.sigmoid(self.fc(x))

            decoder_states += [decoder_state]
            outputs += [output]
            alignments += [ave_alignment]
            dones += [done]


            t += 1
            if test_inputs is None:
                if (done > 0.5).all() and t > self.min_decoder_steps:
                    break
                elif t > self.max_decoder_steps:
                    break

        # Remove 1-element time axis
        #for idx, alignment in enumerate(alignments):
        alignments = list(map(lambda x: x.squeeze(1), alignments))
        decoder_states = list(map(lambda x: x.squeeze(1), decoder_states))
        outputs = list(map(lambda x: x.squeeze(1), outputs))

        # Combine outputs for all time steps
        alignments = torch.stack(alignments).transpose(0, 2)
        #import pdb; pdb.set_trace()
        decoder_states = torch.stack(decoder_states).transpose(0, 1).contiguous()
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()
        outputs = outputs.view(outputs.size(0),-1,self.in_dim)


        return outputs, alignments, dones, decoder_states

    def start_fresh_sequence(self):
        _clear_modules(self.preattention)
        _clear_modules(self.convolutions)
        self.last_conv.clear_buffer()


def _clear_modules(modules):
    for m in modules:
        try:
            m.clear_buffer()
        except AttributeError as e:
            pass


class Converter(nn.Module):
    def __init__(self, n_speakers, speaker_embed_dim,
                 in_dim, out_dim, convolutions=((256, 5, 1),) * 4,
                 time_upsampling=1, r=5,
                 dropout=0.1):
        super(Converter, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_speakers = n_speakers
        self.r = r

        # Non causual convolution blocks
        in_channels = convolutions[0][0]
        # Idea from nyanko
        self.convolutions = nn.ModuleList()

        std_mul = 4.0
        for i, (out_channels, kernel_size, dilation) in enumerate(convolutions):
            '''
            if in_channels != out_channels:
                self.convolutions.append(
                    Conv1d(in_channels, out_channels, kernel_size=1, padding=0,
                           dilation=1, std_mul=std_mul))
                self.convolutions.append(nn.ReLU(inplace=True))
                in_channels = out_channels
                std_mul = 2.0
                
            '''
            if i + 1 == len(convolutions):
                std_mul = 1.0
            self.convolutions.append(
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, out_channels, kernel_size, causal=False,
                          dilation=dilation, dropout=dropout, std_mul=std_mul,
                          residual=True))
            in_channels = out_channels
            #std_mul = 4.0
        # Last fully connect
        self.fc = Linear(in_channels,in_channels*self.r)

        #linear spectrogram fc
        self.linear_spec = Linear(in_channels,self.out_dim)

        #world parameter
        self.upsample = nn.Upsample(scale_factor = time_upsampling)
        self.world = Conv1dGLU(n_speakers, speaker_embed_dim,
                  in_channels, in_channels, kernel_size, causal=False,
                  dilation=dilation, dropout=dropout, std_mul=std_mul,
                  residual=True)
        self.voiced = Linear(in_channels,1)
        self.f0 = Linear(in_channels,1)
        self.sp = Linear(in_channels,513) #hard corded
        self.ap = Linear(in_channels,513)


    def forward(self, x, speaker_embed=None):
        assert self.n_speakers == 1 or speaker_embed is not None

        # expand speaker embedding for all time steps
        speaker_embed_btc = expand_speaker_embed(x, speaker_embed)
        if speaker_embed_btc is not None:
            speaker_embed_btc = F.dropout(speaker_embed_btc, p=self.dropout, training=self.training)

        # Generic case: B x T x C -> B x C x T
        x = x.transpose(1, 2)

        for f in self.convolutions:
            # Case for upsampling
            if speaker_embed_btc is not None and speaker_embed_btc.size(1) != x.size(-1):
                speaker_embed_btc = expand_speaker_embed(x, speaker_embed, tdim=-1)
                speaker_embed_btc = F.dropout(
                    speaker_embed_btc, p=self.dropout, training=self.training)
            x = f(x, speaker_embed_btc) if isinstance(f, Conv1dGLU) else f(x)

        # Back to B x T x C
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = x.view(x.size(0), -1, x.size(-1)//self.r)
        wx = x.transpose(1, 2)

        #linear spectrogram
        x = self.linear_spec(x)

        #world parameter
        wx = self.upsample(wx)
        if speaker_embed_btc is not None and speaker_embed_btc.size(1) != x.size(-1):
            speaker_embed_btc = expand_speaker_embed(x, speaker_embed, tdim=-1)
            speaker_embed_btc = F.dropout(
                speaker_embed_btc, p=self.dropout, training=self.training)
        wx = self.world(wx, speaker_embed_btc)
        wx = wx.transpose(1, 2)
        voiced = torch.sigmoid(self.voiced(wx))
        f0 = self.f0(wx)
        sp = self.sp(wx)
        ap = self.ap(wx)

        return x.view(x.size(0), -1, self.out_dim), voiced[:,:,0], f0[:,:,0], sp, ap
