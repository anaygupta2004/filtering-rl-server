"""
Probe Evaluator for Truth and Deception Detection
==================================================
Evaluates agent responses using pre-trained probes during RL training.
Activation extraction is optional and controlled via parameters.
Probe layers are configurable.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from dataclasses import dataclass, field
import json

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class ProbeScores:
    """Container for probe scores from a single response."""
    # Truth probe scores (dynamic based on config)
    truth_scores: Dict[int, float] = field(default_factory=dict)
    truth_ensemble: float = 0.0
    
    # Deception probe scores (dynamic based on config)
    deception_scores: Dict[int, float] = field(default_factory=dict)
    deception_ensemble: float = 0.0
    
    # Sycophancy probe scores
    sycophancy_scores: Dict[int, float] = field(default_factory=dict)
    sycophancy_ensemble: float = 0.0
    
    # Metadata
    env_name: str = ""
    trajectory_id: str = ""
    subenv_id: str = ""
    reward: float = 0.0
    influence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'truth_scores': self.truth_scores,
            'truth_ensemble': self.truth_ensemble,
            'deception_scores': self.deception_scores,
            'deception_ensemble': self.deception_ensemble,
            'sycophancy_scores': self.sycophancy_scores,
            'sycophancy_ensemble': self.sycophancy_ensemble,
            'env_name': self.env_name,
            'trajectory_id': self.trajectory_id,
            'subenv_id': self.subenv_id,
            'reward': self.reward,
            'influence': self.influence,
        }


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
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
        data = torch.load(path, map_location=device, weights_only=False)
        layer = int(Path(path).stem.split('_')[-1])  # Extract layer from filename
        probe = cls(data['input_dim'], layer=layer)
        probe.load_state_dict(data['state_dict'])
        return probe.to(device)




class SycophancyProbe(nn.Module):
    """Probe for detecting sycophantic behavior."""
    
    def __init__(self, d_model: int, layer: int = 0):
        super().__init__()
        self.d_model = d_model
        self.layer = layer
        self.register_buffer('weights', torch.zeros(d_model))
        self.register_buffer('bias', torch.zeros(1))
        self.register_buffer('mean', torch.zeros(d_model))
        self.register_buffer('std', torch.ones(d_model))
    
    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        if activations.dim() == 3:
            activations = activations[:, -1, :]
        if activations.dim() == 1:
            activations = activations.unsqueeze(0)
        x = (activations - self.mean) / (self.std + 1e-8)
        logits = (x * self.weights).sum(dim=-1) + self.bias
        return torch.sigmoid(logits)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'SycophancyProbe':
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        probe = cls(checkpoint['d_model'], layer=checkpoint['layer'])
        probe.weights = checkpoint['weights'].to(device)
        probe.bias = checkpoint['bias'].to(device)
        probe.mean = checkpoint['mean'].to(device)
        probe.std = checkpoint['std'].to(device)
        return probe.to(device)


class ProbeEvaluator:
    """Unified evaluator for truth and deception probes with configurable layers."""
    
    def __init__(
        self,
        truth_probes: Dict[int, TruthProbe],
        deception_probes: Dict[int, DeceptionProbe],
        sycophancy_probes: Dict[int, SycophancyProbe] = None,
        truth_layers: List[int] = None,
        deception_layers: List[int] = None,
        sycophancy_layers: List[int] = None,
        device: str = 'cuda',
        enabled: bool = True,
    ):
        self.truth_probes = truth_probes
        self.deception_probes = deception_probes
        self.sycophancy_probes = sycophancy_probes or {}
        self.truth_layers = truth_layers or []
        self.deception_layers = deception_layers or []
        self.sycophancy_layers = sycophancy_layers or []
        self.device = device
        self.enabled = enabled
        
        # Move probes to device
        for probe in self.truth_probes.values():
            probe.to(device).eval()
        for probe in self.deception_probes.values():
            probe.to(device).eval()
        for probe in self.sycophancy_probes.values():
            probe.to(device).eval()
        
        # Storage for iteration scores
        self.iteration_scores: List[ProbeScores] = []
        
        print(f"ProbeEvaluator initialized:")
        print(f"  Truth probes: layers {list(self.truth_probes.keys())}")
        print(f"  Deception probes: layers {list(self.deception_probes.keys())}")
        print(f"  Sycophancy probes: layers {list(self.sycophancy_probes.keys())}")
    
    @classmethod
    def from_checkpoints(
        cls,
        truth_probe_dir: Optional[str] = None,
        deception_probe_dir: Optional[str] = None,
        sycophancy_probe_dir: Optional[str] = None,
        truth_layers: Optional[List[int]] = None,
        deception_layers: Optional[List[int]] = None,
        sycophancy_layers: Optional[List[int]] = None,
        device: str = 'cuda',
        enabled: bool = True,
    ) -> 'ProbeEvaluator':
        """
        Load probes from checkpoint directories.
        
        Args:
            truth_probe_dir: Directory containing truth probes
            deception_probe_dir: Directory containing deception probes
            sycophancy_probe_dir: Directory containing sycophancy probes
            truth_layers: Which truth probe layers to load (default: [12, 14, 18])
            deception_layers: Which deception probe layers to load (default: [13, 14, 15])
            sycophancy_layers: Which sycophancy probe layers to load (default: [11, 12, 13])
            device: Device to load probes on
            enabled: Whether probes are enabled
        """
        # Default layers if not specified
        truth_layers = truth_layers or [12, 14, 18]
        deception_layers = deception_layers or [13, 14, 15]
        sycophancy_layers = sycophancy_layers or [11, 12, 13]
        
        truth_probes = {}
        deception_probes = {}
        sycophancy_probes = {}
        
        # Load truth probes
        if truth_probe_dir:
            truth_probe_dir = Path(truth_probe_dir).expanduser()
            for layer in truth_layers:
                path = truth_probe_dir / f"truth_probe_layer{layer}.pt"
                if path.exists():
                    try:
                        truth_probes[layer] = TruthProbe.load(str(path), device)
                        print(f"Loaded truth probe layer {layer}")
                    except Exception as e:
                        print(f"Warning: Failed to load truth probe layer {layer}: {e}")
        
        # Load deception probes
        if deception_probe_dir:
            deception_probe_dir = Path(deception_probe_dir).expanduser()
            for layer in deception_layers:
                path = deception_probe_dir / f"probe_layer_{layer}.pt"
                if path.exists():
                    try:
                        deception_probes[layer] = DeceptionProbe.load_pytorch(str(path), device)
                        print(f"Loaded deception probe layer {layer}")
                    except Exception as e:
                        print(f"Warning: Failed to load deception probe layer {layer}: {e}")
        
        # Load sycophancy probes
        if sycophancy_probe_dir:
            sycophancy_probe_dir = Path(sycophancy_probe_dir).expanduser()
            for layer in sycophancy_layers:
                path = sycophancy_probe_dir / f"sycophancy_probe_layer_{layer}.pt"
                if path.exists():
                    try:
                        sycophancy_probes[layer] = SycophancyProbe.load(str(path), device)
                        print(f"Loaded sycophancy probe layer {layer}")
                    except Exception as e:
                        print(f"Warning: Failed to load sycophancy probe layer {layer}: {e}")
        
        return cls(
            truth_probes=truth_probes,
            deception_probes=deception_probes,
            sycophancy_probes=sycophancy_probes,
            truth_layers=truth_layers,
            deception_layers=deception_layers,
            sycophancy_layers=sycophancy_layers,
            device=device,
            enabled=enabled,
        )
    
    def get_required_layers(self) -> List[int]:
        """Return list of layers needed for probe evaluation."""
        if not self.enabled:
            return []
        layers = set()
        layers.update(self.truth_probes.keys())
        layers.update(self.deception_probes.keys())
        layers.update(self.sycophancy_probes.keys())
        return sorted(layers)
    
    @torch.no_grad()
    def evaluate_activations(
        self,
        activations_dict: Dict[int, torch.Tensor],
        env_name: str = "",
        trajectory_id: str = "",
        subenv_id: str = "",
        reward: float = 0.0,
        influence: float = 0.0,
    ) -> ProbeScores:
        """
        Evaluate activations using all probes.
        
        Args:
            activations_dict: {layer: activations} where activations are [hidden_dim] or [batch, hidden_dim]
            env_name: Environment name for logging
            trajectory_id: Trajectory ID for logging
            subenv_id: Sub-environment ID
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
            subenv_id=subenv_id,
            reward=reward,
            influence=influence,
        )
        
        # Truth probes
        truth_scores_list = []
        for layer, probe in self.truth_probes.items():
            if layer in activations_dict:
                acts = activations_dict[layer]
                if isinstance(acts, np.ndarray):
                    acts = torch.from_numpy(acts)
                acts = acts.to(self.device)
                score = probe(acts).mean().item()
                scores.truth_scores[layer] = score
                truth_scores_list.append(score)
        
        if truth_scores_list:
            scores.truth_ensemble = float(np.mean(truth_scores_list))
        
        # Deception probes
        deception_scores_list = []
        for layer, probe in self.deception_probes.items():
            if layer in activations_dict:
                acts = activations_dict[layer]
                if isinstance(acts, np.ndarray):
                    acts = torch.from_numpy(acts)
                acts = acts.to(self.device)
                score = probe(acts).mean().item()
                scores.deception_scores[layer] = score
                deception_scores_list.append(score)
        
        if deception_scores_list:
            scores.deception_ensemble = float(np.mean(deception_scores_list))
        
        # Sycophancy probes
        sycophancy_scores_list = []
        for layer, probe in self.sycophancy_probes.items():
            if layer in activations_dict:
                acts = activations_dict[layer]
                if isinstance(acts, np.ndarray):
                    acts = torch.from_numpy(acts)
                acts = acts.to(self.device)
                score = probe(acts).mean().item()
                scores.sycophancy_scores[layer] = score
                sycophancy_scores_list.append(score)
        
        if sycophancy_scores_list:
            scores.sycophancy_ensemble = float(np.mean(sycophancy_scores_list))
        
        return scores
    
    def add_score(self, score: ProbeScores):
        """Add a score to the iteration buffer."""
        self.iteration_scores.append(score)
    
    def clear_scores(self):
        """Clear the iteration score buffer."""
        self.iteration_scores = []
    
    def get_scores_summary(self) -> Dict[str, Any]:
        """Get summary of collected scores."""
        if not self.iteration_scores:
            return {}
        
        return {
            'num_scores': len(self.iteration_scores),
            'truth_mean': np.mean([s.truth_ensemble for s in self.iteration_scores]),
            'deception_mean': np.mean([s.deception_ensemble for s in self.iteration_scores]),
        }
    
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
        sycophancy_ensemble = np.array([s.sycophancy_ensemble for s in self.iteration_scores])
        rewards = np.array([s.reward for s in self.iteration_scores])
        influences = np.array([s.influence for s in self.iteration_scores])
        
        # Aggregate metrics
        metrics["probe/truth_mean"] = float(np.mean(truth_ensemble))
        metrics["probe/truth_std"] = float(np.std(truth_ensemble))
        metrics["probe/truth_min"] = float(np.min(truth_ensemble))
        metrics["probe/truth_max"] = float(np.max(truth_ensemble))
        
        metrics["probe/deception_mean"] = float(np.mean(deception_ensemble))
        metrics["probe/deception_std"] = float(np.std(deception_ensemble))
        metrics["probe/deception_min"] = float(np.min(deception_ensemble))
        metrics["probe/deception_max"] = float(np.max(deception_ensemble))
        
        metrics["probe/sycophancy_mean"] = float(np.mean(sycophancy_ensemble))
        metrics["probe/sycophancy_std"] = float(np.std(sycophancy_ensemble))
        metrics["probe/sycophancy_min"] = float(np.min(sycophancy_ensemble))
        metrics["probe/sycophancy_max"] = float(np.max(sycophancy_ensemble))
        
        # Per-layer metrics
        for layer in self.truth_probes.keys():
            vals = [s.truth_scores.get(layer, 0) for s in self.iteration_scores]
            if vals:
                metrics[f"probe/truth_L{layer}_mean"] = float(np.mean(vals))
        
        for layer in self.deception_probes.keys():
            vals = [s.deception_scores.get(layer, 0) for s in self.iteration_scores]
            if vals:
                metrics[f"probe/deception_L{layer}_mean"] = float(np.mean(vals))
        
        for layer in self.sycophancy_probes.keys():
            vals = [s.sycophancy_scores.get(layer, 0) for s in self.iteration_scores]
            if vals:
                metrics[f"probe/sycophancy_L{layer}_mean"] = float(np.mean(vals))
        
        # Chosen vs rejected
        if chosen_indices and rejected_indices:
            n_scores = len(self.iteration_scores)
            chosen_truth = [self.iteration_scores[i].truth_ensemble for i in chosen_indices if i < n_scores]
            rejected_truth = [self.iteration_scores[i].truth_ensemble for i in rejected_indices if i < n_scores]
            chosen_deception = [self.iteration_scores[i].deception_ensemble for i in chosen_indices if i < n_scores]
            rejected_deception = [self.iteration_scores[i].deception_ensemble for i in rejected_indices if i < n_scores]
            chosen_sycophancy = [self.iteration_scores[i].sycophancy_ensemble for i in chosen_indices if i < n_scores]
            rejected_sycophancy = [self.iteration_scores[i].sycophancy_ensemble for i in rejected_indices if i < n_scores]
            
            if chosen_truth:
                metrics["probe/truth_chosen_mean"] = float(np.mean(chosen_truth))
                metrics["probe/deception_chosen_mean"] = float(np.mean(chosen_deception))
                metrics["probe/sycophancy_chosen_mean"] = float(np.mean(chosen_sycophancy))
            if rejected_truth:
                metrics["probe/truth_rejected_mean"] = float(np.mean(rejected_truth))
                metrics["probe/deception_rejected_mean"] = float(np.mean(rejected_deception))
                metrics["probe/sycophancy_rejected_mean"] = float(np.mean(rejected_sycophancy))
        
        # Correlations
        if len(rewards) > 1 and np.std(rewards) > 1e-6 and np.std(truth_ensemble) > 1e-6:
            metrics["probe/truth_reward_corr"] = float(np.corrcoef(truth_ensemble, rewards)[0, 1])
        if len(influences) > 1 and np.std(influences) > 1e-6 and np.std(deception_ensemble) > 1e-6:
            metrics["probe/deception_influence_corr"] = float(np.corrcoef(deception_ensemble, influences)[0, 1])
        
        # Per-environment breakdown
        env_names = set(s.env_name for s in self.iteration_scores if s.env_name)
        for env_name in env_names:
            env_scores = [s for s in self.iteration_scores if s.env_name == env_name]
            if env_scores:
                metrics[f"probe/truth_{env_name}"] = float(np.mean([s.truth_ensemble for s in env_scores]))
                metrics[f"probe/deception_{env_name}"] = float(np.mean([s.deception_ensemble for s in env_scores]))
                metrics[f"probe/sycophancy_{env_name}"] = float(np.mean([s.sycophancy_ensemble for s in env_scores]))
        
        return metrics
    
    def log_to_wandb(
        self,
        chosen_indices: Optional[List[int]] = None,
        rejected_indices: Optional[List[int]] = None,
        step: Optional[int] = None,
    ) -> Dict[str, float]:
        """Log iteration metrics to wandb."""
        metrics = self.compute_iteration_metrics(chosen_indices, rejected_indices)
        
        if metrics and WANDB_AVAILABLE:
            # Add histograms
            truth_vals = [s.truth_ensemble for s in self.iteration_scores]
            deception_vals = [s.deception_ensemble for s in self.iteration_scores]
            
            if truth_vals:
                metrics["probe/truth_histogram"] = wandb.Histogram(truth_vals)
            if deception_vals:
                metrics["probe/deception_histogram"] = wandb.Histogram(deception_vals)
            sycophancy_vals = [s.sycophancy_ensemble for s in self.iteration_scores]
            if sycophancy_vals:
                metrics["probe/sycophancy_histogram"] = wandb.Histogram(sycophancy_vals)
            
            wandb.log(metrics, step=step)
        
        # Clear for next iteration
        self.clear_scores()
        
        return metrics
    
    def save_scores(self, path: str):
        """Save collected scores to JSON."""
        data = [s.to_dict() for s in self.iteration_scores]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
