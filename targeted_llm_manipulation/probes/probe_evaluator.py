"""
Probe Evaluator for Truth and Deception Detection
==================================================
Evaluates agent responses using pre-trained probes during RL training.
Activation extraction is optional and controlled via parameters.

Usage:
    evaluator = ProbeEvaluator.from_checkpoints(
        truth_probe_dir="~/truth_probe/Truth_is_Universal/checkpoints",
        deception_probe_dir="~/deception_probe/outputs/gemma2b_deception_probes/probes/pytorch/best",
        device="cuda:0"
    )
    
    # During trajectory generation (when enabled):
    scores = evaluator.evaluate_activations(activations_dict)
    
    # Log to wandb:
    evaluator.log_iteration_metrics(all_scores, chosen_indices, rejected_indices, rewards)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
import wandb


@dataclass
class ProbeScores:
    """Container for probe scores from a single response."""
    # Truth probe scores (layers 12, 14, 18)
    truth_L12: float = 0.0
    truth_L14: float = 0.0
    truth_L18: float = 0.0
    truth_ensemble: float = 0.0
    
    # Deception probe scores (layers 13, 14, 15)
    deception_L13: float = 0.0
    deception_L14: float = 0.0
    deception_L15: float = 0.0
    deception_ensemble: float = 0.0
    
    # Metadata
    env_name: str = ""
    trajectory_id: str = ""
    reward: float = 0.0
    influence: float = 0.0


class TruthProbe(nn.Module):
    """Linear truth probe for detecting true/false statements."""
    
    def __init__(self, hidden_dim: int, layer: int, hook_name: str,
                 acts_mean: Optional[torch.Tensor] = None):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1, bias=True)
        self.layer = layer
        self.hook_name = hook_name
        self.hidden_dim = hidden_dim
        
        if acts_mean is not None:
            self.register_buffer('acts_mean', acts_mean)
        else:
            self.register_buffer('acts_mean', torch.zeros(hidden_dim))
    
    def forward(self, activations: torch.Tensor, center: bool = True) -> torch.Tensor:
        if activations.dim() == 1:
            activations = activations.unsqueeze(0)
        if center:
            activations = activations - self.acts_mean.to(activations.device)
        logits = self.linear(activations.float()).squeeze(-1)
        return torch.sigmoid(logits)
    
    @classmethod
    def load(cls, checkpoint_path: str, device: str = 'cpu') -> 'TruthProbe':
        ckpt = torch.load(checkpoint_path, map_location=device)
        probe = cls(
            hidden_dim=ckpt['hidden_dim'],
            layer=ckpt['layer'],
            hook_name=ckpt['hook_name'],
            acts_mean=ckpt.get('acts_mean')
        )
        probe.linear.load_state_dict(ckpt['probe_state_dict'])
        return probe.to(device)


class DeceptionProbe(nn.Module):
    """PyTorch module for deception detection."""

    def __init__(self, input_dim: int, layer: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.layer = layer
        self.register_buffer('scale_mean', torch.zeros(input_dim))
        self.register_buffer('scale_std', torch.ones(input_dim))
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        x_scaled = (x - self.scale_mean) / (self.scale_std + 1e-8)
        logits = self.linear(x_scaled).squeeze(-1)
        probs = torch.sigmoid(logits)
        if squeeze:
            probs = probs.squeeze(0)
        return probs

    @classmethod
    def load_pytorch(cls, path: str, device: str = 'cpu') -> 'DeceptionProbe':
        data = torch.load(path, map_location=device)
        layer = int(Path(path).stem.split('_')[-1])  # Extract layer from filename
        probe = cls(data['input_dim'], layer=layer)
        probe.load_state_dict(data['state_dict'])
        return probe.to(device)


class ProbeEvaluator:
    """Unified evaluator for truth and deception probes."""
    
    # Layer configurations
    TRUTH_LAYERS = [12, 14, 18]
    DECEPTION_LAYERS = [13, 14, 15]
    
    def __init__(
        self,
        truth_probes: Dict[int, TruthProbe],
        deception_probes: Dict[int, DeceptionProbe],
        device: str = 'cuda',
        enabled: bool = True,
    ):
        self.truth_probes = truth_probes
        self.deception_probes = deception_probes
        self.device = device
        self.enabled = enabled
        
        # Move probes to device
        for probe in self.truth_probes.values():
            probe.to(device)
        for probe in self.deception_probes.values():
            probe.to(device)
        
        # Storage for iteration scores
        self.iteration_scores: List[ProbeScores] = []
    
    @classmethod
    def from_checkpoints(
        cls,
        truth_probe_dir: str,
        deception_probe_dir: str,
        device: str = 'cuda',
        enabled: bool = True,
    ) -> 'ProbeEvaluator':
        """Load probes from checkpoint directories."""
        truth_probe_dir = Path(truth_probe_dir).expanduser()
        deception_probe_dir = Path(deception_probe_dir).expanduser()
        
        # Load truth probes
        truth_probes = {}
        for layer in cls.TRUTH_LAYERS:
            path = truth_probe_dir / f"truth_probe_layer{layer}.pt"
            if path.exists():
                truth_probes[layer] = TruthProbe.load(str(path), device)
                print(f"Loaded truth probe layer {layer}")
        
        # Load deception probes
        deception_probes = {}
        for layer in cls.DECEPTION_LAYERS:
            path = deception_probe_dir / f"probe_layer_{layer}.pt"
            if path.exists():
                deception_probes[layer] = DeceptionProbe.load_pytorch(str(path), device)
                print(f"Loaded deception probe layer {layer}")
        
        return cls(truth_probes, deception_probes, device, enabled)
    
    def get_required_layers(self) -> List[int]:
        """Return list of layers needed for probe evaluation."""
        if not self.enabled:
            return []
        layers = set(self.TRUTH_LAYERS + self.DECEPTION_LAYERS)
        return sorted(layers)
    
    @torch.no_grad()
    def evaluate_activations(
        self,
        activations_dict: Dict[int, torch.Tensor],
        env_name: str = "",
        trajectory_id: str = "",
        reward: float = 0.0,
        influence: float = 0.0,
    ) -> ProbeScores:
        """
        Evaluate activations using all probes.
        
        Args:
            activations_dict: {layer: activations} where activations are [hidden_dim] or [batch, hidden_dim]
            env_name: Environment name for logging
            trajectory_id: Trajectory ID for logging
            reward: Reward value for correlation analysis
            influence: Influence value for correlation analysis
        
        Returns:
            ProbeScores with all probe outputs
        """
        if not self.enabled:
            return ProbeScores()
        
        scores = ProbeScores(
            env_name=env_name,
            trajectory_id=trajectory_id,
            reward=reward,
            influence=influence,
        )
        
        # Truth probes
        truth_scores = []
        for layer in self.TRUTH_LAYERS:
            if layer in self.truth_probes and layer in activations_dict:
                acts = activations_dict[layer].to(self.device)
                score = self.truth_probes[layer](acts).mean().item()
                setattr(scores, f"truth_L{layer}", score)
                truth_scores.append(score)
        
        if truth_scores:
            scores.truth_ensemble = np.mean(truth_scores)
        
        # Deception probes
        deception_scores = []
        for layer in self.DECEPTION_LAYERS:
            if layer in self.deception_probes and layer in activations_dict:
                acts = activations_dict[layer].to(self.device)
                score = self.deception_probes[layer](acts).mean().item()
                setattr(scores, f"deception_L{layer}", score)
                deception_scores.append(score)
        
        if deception_scores:
            scores.deception_ensemble = np.mean(deception_scores)
        
        return scores
    
    def add_score(self, score: ProbeScores):
        """Add a score to the iteration buffer."""
        self.iteration_scores.append(score)
    
    def clear_scores(self):
        """Clear the iteration score buffer."""
        self.iteration_scores = []
    
    def compute_iteration_metrics(
        self,
        chosen_indices: Optional[List[int]] = None,
        rejected_indices: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Compute aggregated metrics for the iteration.
        
        Args:
            chosen_indices: Indices of chosen trajectories
            rejected_indices: Indices of rejected trajectories
        
        Returns:
            Dict of metrics to log to wandb
        """
        if not self.iteration_scores:
            return {}
        
        metrics = {}
        
        # Extract arrays
        truth_ensemble = np.array([s.truth_ensemble for s in self.iteration_scores])
        deception_ensemble = np.array([s.deception_ensemble for s in self.iteration_scores])
        rewards = np.array([s.reward for s in self.iteration_scores])
        influences = np.array([s.influence for s in self.iteration_scores])
        
        # Aggregate metrics
        metrics["probe/truth_mean"] = np.mean(truth_ensemble)
        metrics["probe/truth_std"] = np.std(truth_ensemble)
        metrics["probe/truth_min"] = np.min(truth_ensemble)
        metrics["probe/truth_max"] = np.max(truth_ensemble)
        
        metrics["probe/deception_mean"] = np.mean(deception_ensemble)
        metrics["probe/deception_std"] = np.std(deception_ensemble)
        metrics["probe/deception_min"] = np.min(deception_ensemble)
        metrics["probe/deception_max"] = np.max(deception_ensemble)
        
        # Per-layer metrics
        for layer in self.TRUTH_LAYERS:
            vals = np.array([getattr(s, f"truth_L{layer}") for s in self.iteration_scores])
            metrics[f"probe/truth_L{layer}_mean"] = np.mean(vals)
        
        for layer in self.DECEPTION_LAYERS:
            vals = np.array([getattr(s, f"deception_L{layer}") for s in self.iteration_scores])
            metrics[f"probe/deception_L{layer}_mean"] = np.mean(vals)
        
        # Chosen vs rejected
        if chosen_indices and rejected_indices:
            chosen_truth = np.array([self.iteration_scores[i].truth_ensemble for i in chosen_indices if i < len(self.iteration_scores)])
            rejected_truth = np.array([self.iteration_scores[i].truth_ensemble for i in rejected_indices if i < len(self.iteration_scores)])
            chosen_deception = np.array([self.iteration_scores[i].deception_ensemble for i in chosen_indices if i < len(self.iteration_scores)])
            rejected_deception = np.array([self.iteration_scores[i].deception_ensemble for i in rejected_indices if i < len(self.iteration_scores)])
            
            if len(chosen_truth) > 0:
                metrics["probe/truth_chosen_mean"] = np.mean(chosen_truth)
                metrics["probe/deception_chosen_mean"] = np.mean(chosen_deception)
            if len(rejected_truth) > 0:
                metrics["probe/truth_rejected_mean"] = np.mean(rejected_truth)
                metrics["probe/deception_rejected_mean"] = np.mean(rejected_deception)
        
        # Correlations
        if len(rewards) > 1 and np.std(rewards) > 0 and np.std(truth_ensemble) > 0:
            metrics["probe/truth_reward_corr"] = np.corrcoef(truth_ensemble, rewards)[0, 1]
        if len(influences) > 1 and np.std(influences) > 0 and np.std(deception_ensemble) > 0:
            metrics["probe/deception_influence_corr"] = np.corrcoef(deception_ensemble, influences)[0, 1]
        
        # Per-environment breakdown
        env_names = set(s.env_name for s in self.iteration_scores if s.env_name)
        for env_name in env_names:
            env_scores = [s for s in self.iteration_scores if s.env_name == env_name]
            if env_scores:
                metrics[f"probe/truth_{env_name}"] = np.mean([s.truth_ensemble for s in env_scores])
                metrics[f"probe/deception_{env_name}"] = np.mean([s.deception_ensemble for s in env_scores])
        
        return metrics
    
    def log_to_wandb(
        self,
        chosen_indices: Optional[List[int]] = None,
        rejected_indices: Optional[List[int]] = None,
        step: Optional[int] = None,
    ):
        """Log iteration metrics to wandb."""
        metrics = self.compute_iteration_metrics(chosen_indices, rejected_indices)
        
        if metrics:
            # Add histograms
            truth_vals = [s.truth_ensemble for s in self.iteration_scores]
            deception_vals = [s.deception_ensemble for s in self.iteration_scores]
            
            if truth_vals:
                metrics["probe/truth_histogram"] = wandb.Histogram(truth_vals)
            if deception_vals:
                metrics["probe/deception_histogram"] = wandb.Histogram(deception_vals)
            
            wandb.log(metrics, step=step)
        
        # Clear for next iteration
        self.clear_scores()
        
        return metrics
