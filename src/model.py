# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jingyu Wang
# Decision Transformer attribution and upstream notice: ../THIRD_PARTY_NOTICES.md
"""
Decision Transformer and SASRec for next-genre prediction.

The Decision Transformer portion was documented as adapted from a MuJoCo
Decision Transformer implementation. See THIRD_PARTY_NOTICES.md for the
official reference implementation, its MIT notice, and provenance details.
SASRec follows Kang and McAuley (2018), also credited there.

States summarize prior genre preferences and ratings. Actions are discrete
genres, rewards are normalized ratings, and RTG sums rewards over a forward
window of 20 interactions in the paper configurations. The action head uses
cross-entropy instead of continuous-action regression.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCausalAttention(nn.Module):
    def __init__(self, hidden_size, n_head, dropout=0.1):
        super().__init__()
        assert hidden_size % n_head == 0
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        causal = torch.tril(
            torch.ones(T, T, dtype=torch.bool, device=x.device)
        ).unsqueeze(0).unsqueeze(0)
        allowed = causal
        if attention_mask is not None:
            valid = attention_mask.to(device=x.device, dtype=torch.bool)
            if valid.shape != (B, T):
                raise ValueError(
                    f"attention_mask must have shape {(B, T)}, got {tuple(valid.shape)}"
                )
            # Real queries may attend only to real causal keys. Give padded
            # queries a self-edge to avoid an all-masked softmax row; padded
            # keys remain invisible to every real query.
            key_valid = valid[:, None, None, :]
            query_valid = valid[:, None, :, None]
            padded_self = (
                ~query_valid
                & torch.eye(T, dtype=torch.bool, device=x.device)[None, None, :, :]
            )
            allowed = (causal & key_valid) | padded_self
        attn = attn.masked_fill(~allowed, float('-inf'))
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, n_head, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.attn = MaskedCausalAttention(hidden_size, n_head, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class SASRec(nn.Module):
    """
    SASRec: Self-Attentive Sequential Recommendation.
    Input: sequence of category IDs only — no state features, no RTG.
    Structurally cannot condition on a target return at inference time.

    use_pad_token: if True, the embedding table has one extra row at index
    n_actions used as a dedicated PAD token. The dataset must left-pad with
    this index. Removes the architectural smell of using action=0 (a real
    genre) to also mean "padding."
    """

    def __init__(
        self,
        n_actions: int,
        hidden_size: int = 128,
        max_length: int = 20,
        n_layer: int = 2,
        n_head: int = 4,
        dropout: float = 0.2,
        use_pad_token: bool = False,
    ):
        super().__init__()
        self.use_pad_token = use_pad_token
        self.n_actions = n_actions
        embed_size = n_actions + (1 if use_pad_token else 0)
        self.item_embed  = nn.Embedding(embed_size, hidden_size)
        self.pos_embed   = nn.Embedding(max_length, hidden_size)
        self.embed_ln    = nn.LayerNorm(hidden_size)
        self.drop        = nn.Dropout(dropout)
        self.blocks      = nn.ModuleList([
            TransformerBlock(hidden_size, n_head, dropout) for _ in range(n_layer)
        ])
        self.ln_out      = nn.LayerNorm(hidden_size)
        self.action_head = nn.Linear(hidden_size, n_actions)  # never predict PAD

    def forward(self, actions, attention_mask=None):
        """actions: (B, T) long — category indices"""
        B, T = actions.shape
        if attention_mask is None and self.use_pad_token:
            attention_mask = actions.ne(self.n_actions)
        pos = torch.arange(T, device=actions.device).unsqueeze(0)
        x   = self.drop(self.embed_ln(self.item_embed(actions) + self.pos_embed(pos)))
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        return self.action_head(self.ln_out(x))  # (B, T, n_actions)


class RecDecisionTransformer(nn.Module):
    """
    Decision Transformer for recommendation.

    At each step t, the context is a sequence of (RTG, state, action) tuples.
    State is a feature vector of the user's current context.
    Action is a discrete item category (system decision).
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,          # number of item categories
        hidden_size: int = 128,
        max_length: int = 20,    # context window K
        n_layer: int = 3,
        n_head: int = 1,
        dropout: float = 0.1,
        use_rtg: bool = True,    # set False for DT-no-RTG baseline
        use_pad_token: bool = False,  # add a dedicated PAD row to action_embed
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.n_actions = n_actions
        self.use_rtg = use_rtg
        self.use_pad_token = use_pad_token

        # Input embeddings
        self.state_embed = nn.Linear(state_dim, hidden_size)
        action_embed_size = n_actions + (1 if use_pad_token else 0)
        self.action_embed = nn.Embedding(action_embed_size, hidden_size)
        self.rtg_embed = nn.Linear(1, hidden_size)

        # Positional embedding over context window (3 tokens per step: RTG, s, a)
        self.pos_embed = nn.Embedding(3 * max_length, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, n_head, dropout) for _ in range(n_layer)
        ])
        self.ln_out = nn.LayerNorm(hidden_size)

        # Action prediction head (discrete — cross entropy)
        self.action_head = nn.Linear(hidden_size, n_actions)

    def forward(
        self,
        states,
        actions,
        returns_to_go,
        timesteps=None,
        attention_mask=None,
    ):
        """
        states:        (B, T, state_dim)
        actions:       (B, T) — long, category indices
        returns_to_go: (B, T, 1)
        timesteps:     (B, T) — long, accepted for API compatibility but unused.
                       This model uses relative positional encoding (0 … 3T-1) rather
                       than absolute timestep embeddings, so the argument is ignored.

        Returns action logits: (B, T, n_actions)
        """
        B, T, _ = states.shape

        if attention_mask is None and self.use_pad_token:
            # Padding uses zero state and zero RTG. The current unknown action
            # also uses PAD, so action ID alone cannot define validity.
            attention_mask = (
                states.abs().sum(dim=-1).ne(0)
                | returns_to_go.abs().sum(dim=-1).ne(0)
            )

        s_emb = self.state_embed(states)           # (B, T, H)
        a_emb = self.action_embed(actions)          # (B, T, H)

        if self.use_rtg:
            r_emb = self.rtg_embed(returns_to_go)  # (B, T, H)
        else:
            r_emb = torch.zeros_like(s_emb)        # zero out RTG for baseline

        # Interleave: [RTG_1, s_1, a_1, RTG_2, s_2, a_2, ...]
        # Shape: (B, 3T, H)
        seq = torch.stack([r_emb, s_emb, a_emb], dim=2).reshape(B, 3 * T, self.hidden_size)

        pos = torch.arange(3 * T, device=states.device).unsqueeze(0)
        seq = self.drop(self.embed_ln(seq + self.pos_embed(pos)))

        token_attention_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (B, T):
                raise ValueError(
                    f"attention_mask must have shape {(B, T)}, "
                    f"got {tuple(attention_mask.shape)}"
                )
            token_attention_mask = attention_mask.repeat_interleave(3, dim=1)

        for block in self.blocks:
            seq = block(seq, attention_mask=token_attention_mask)
        seq = self.ln_out(seq)

        # State tokens are at positions 1, 4, 7, ... (index 3t+1)
        # Predict action from state token
        state_tokens = seq[:, 1::3]                # (B, T, H)
        logits = self.action_head(state_tokens)    # (B, T, n_actions)
        return logits
