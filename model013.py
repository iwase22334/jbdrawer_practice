import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, d_model, d_vocab, num_heads, dropout=0.1, device='cpu'):
        super().__init__()
        self.d_model = d_model
        self.d_vocab = d_vocab
        self.device = device

        self.mh_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.ln = nn.LayerNorm([d_model])

        self.seq = nn.Sequential(
            nn.Linear(self.d_model, d_model),
            nn.ReLU(),
        )

        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x_mask):
        x, mask = x_mask

        x = x.masked_fill(mask, 0)
        # kpm = torch.where(mask == 1, torch.tensor(-float(1e9)), mask)
        kpm = mask.type(torch.float32) * -1e9
        # (B, 2*seq_len, d_model) -> (B, 2*seq_len, d_model)
        y, _ = self.mh_attention(x, x, x, key_padding_mask=kpm)
        y = y.masked_fill(mask, 0)

        # (B, 2*seq_len, d_model) -> (B, 2*seq_len, d_model)
        ln_y = self.ln(y + x)

        # (B, 2*seq_len, d_model) -> (B, 2*seq_len, d_model)
        ff_y = self.seq(ln_y)
        y = self.ln2(ln_y + ff_y)

        return (y, mask)


class DecisionTransformer(nn.Module):
    def __init__(self, param):
        super().__init__()

        self.d_model = param["d_model"]
        self.d_vocab = param["d_vocab"]
        self.seq_len = param["seq_len"]

        self.state_encoder = nn.Sequential(  # (B, 2, 128, 128) -> (B, 32, 31, 31)
                                             nn.Conv2d(2, 32, 8, stride=4, padding=0), nn.ReLU(),
                                             # (B, 32, 31, 31) -> (B, 64, 14, 14)
                                             nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                             # (B, 64, 14, 14) -> (B, 64, 12, 12)
                                             nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
                                             # (B, 64, 12, 12) -> (B, 64*12*12)
                                             nn.Flatten(),
                                             nn.Linear(64 * 12 * 12, self.d_model),
                                             nn.Tanh())
        self.action_embeddings = nn.Sequential(nn.Embedding(self.d_vocab, self.d_model),
                                               nn.Tanh())

        self.pos_encoder = PositionalEncoding(self.d_model, max_len=param["seq_len"])
        self.embed_timestep = nn.Embedding(self.seq_len, self.d_model)
        self.embed_ln = nn.LayerNorm(self.d_model)

        self.tf = nn.Sequential(*[Transformer(param["d_model"], param["d_vocab"], num_heads=param["n_head"])
                                for _ in range(param["n_layer"])])

        self.predict_q = torch.nn.Linear(self.d_model, self.d_vocab)

    def forward(self, states, actions, timesteps, mask, cstep):
        batch_size, seq_len = states.shape[0], states.shape[1]

        # to make the attention mask fit the stacked inp, actions, timestepsuts, have to stack it as well
        stacked_attention_mask = torch.stack((mask, mask), dim=1)
        stacked_attention_mask = stacked_attention_mask.permute(0, 2, 1)
        stacked_attention_mask = stacked_attention_mask.reshape(batch_size, 2 * seq_len)
        stacked_attention_mask = stacked_attention_mask.transpose(0, 1)

        # (B, seq_len, 2, 64, 64) -> (B x seq_len, 2, 64, 64)
        stacked_states = states.reshape(-1, 2, 128, 128).type(torch.float32).contiguous()
        state_embeddings = self.state_encoder(stacked_states)
        state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1], self.d_model)

        # (B, seq_len, 1) -> (B, seq_len, d_model)
        action_embeddings = self.action_embeddings(actions)
        action_embeddings = action_embeddings.squeeze(2)

        # (B, seq_len, 1) -> (B, seq_len, d_model)
        time_embeddings = self.embed_timestep(timesteps)
        time_embeddings = time_embeddings.squeeze(2)

        # time embeddings are treated similar to positional embeddings
        # (B, seq_len, d_model) -> (B, seq_len, d_model)
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions

        # (B, seq_len, d_model) -> (B, 2, seq_len, d_model)
        stacked_inputs = torch.stack(
            (state_embeddings, action_embeddings), dim=1
        )
        # (B, 2, seq_len, d_model) -> (B, seq_len, 2, d_model)
        stacked_inputs = stacked_inputs.permute(0, 2, 1, 3)
        # (B, seq_len, 2, d_model) -> (B, 2*seq_len, d_model)
        stacked_inputs = stacked_inputs.reshape(batch_size, 2 * seq_len, self.d_model)
        # (B, 2*seq_len, d_model) -> (B, 2*seq_len, d_model)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        x, _ = self.tf((stacked_inputs, stacked_attention_mask))

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # (B, 2*seq_len, d_model) -> (B, seq_len, 2, d_model)
        x = x.reshape(batch_size, seq_len, 2, self.d_model)
        # (B, seq_len, 2, d_model) -> (B, 2, seq_len, d_model)
        x = x.permute(0, 2, 1, 3)

        # get predictions
        # (B, 2, seq_len, d_model) -> (B, 1, seq_len, vocab)
        q_preds = self.predict_q(x[:, 0])  # predict next action given state

        # (B, 1, seq_len, vocab) -> (B, vocab)
        q_preds = q_preds[:, cstep]
        return q_preds
