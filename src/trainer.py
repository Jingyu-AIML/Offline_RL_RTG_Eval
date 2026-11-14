"""
Training steps for Decision Transformer and SASRec.
"""

import torch
import torch.nn as nn


class RecDTTrainer:
    """Trainer for RecDecisionTransformer."""

    def __init__(self, model, optimizer, device='cpu', scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.loss_fn = nn.CrossEntropyLoss()

    def train_step(self, batch):
        states, actions, rtgs, timesteps, targets = [x.to(self.device) for x in batch]
        self.model.train()
        logits = self.model(states, actions, rtgs, timesteps)  # (B, T, n_actions)
        # Predict action at last timestep
        logits_last = logits[:, -1, :]  # (B, n_actions)
        loss = self.loss_fn(logits_last, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        return loss.item()

    def train_epoch(self, loader):
        total_loss, n = 0.0, 0
        for batch in loader:
            loss = self.train_step(batch)
            total_loss += loss
            n += 1
        return total_loss / n

class SASRecTrainer:
    """Trainer for SASRec baseline."""

    def __init__(self, model, optimizer, device='cpu', scheduler=None):
        self.model     = model
        self.optimizer = optimizer
        self.device    = device
        self.scheduler = scheduler
        self.loss_fn   = nn.CrossEntropyLoss()

    def train_step(self, batch):
        actions, targets = batch[0].to(self.device), batch[1].to(self.device)
        self.model.train()
        logits = self.model(actions)       # (B, T, n_actions)
        loss   = self.loss_fn(logits[:, -1, :], targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        return loss.item()

    def train_epoch(self, loader):
        total_loss, n = 0.0, 0
        for batch in loader:
            total_loss += self.train_step(batch)
            n += 1
        return total_loss / n
