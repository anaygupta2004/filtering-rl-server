"""
Probe Evaluator for Truth and Deception Detection
==================================================
Evaluates agent responses using pre-trained probes during RL training.
Supports multi-position evaluation: first, last, avg, median, max, min token positions.
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

# Token positions to evaluate
POSITIONS = ['first', 'last', 'avg', 'median', 'max', 'min']


@dataclass
class ProbeScores:
    """Container for probe scores from a single response with multi-position support."""
    # Truth probe scores: {layer: {position: score}}
    truth_scores: Dict[int, Dict[str, float]] = field(default_factory=dict)
    truth_ensemble: Dict[str, float] = field(default_factory=dict)  # {position: ensemble_score}
    
    # Deception probe scores: {layer: {position: score}}
    deception_scores: Dict[int, Dict[str, float]] = field(default_factory=dict)
    deception_ensemble: Dict[str, float] = field(default_factory=dict)
    
    # Sycophancy probe scores: {layer: {position: score}}
    sycophancy_scores: Dict[int, Dict[str, float]] = field(default_factory=dict)
    sycophancy_ensemble: Dict[str, float] = field(default_factory=dict)
    
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
    """Unified evaluator for truth and deception probes with multi-position support."""
    
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
        truth_layers = truth_layers or [12, 14, 18]
        deception_layers = deception_layers or [13, 14, 15]
        sycophancy_layers = sycophancy_layers or [11, 12, 13]
        
        truth_probes = {}
        deception_probes = {}
        sycophancy_probes = {}
        
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
    
    def _compute_scores_for_all_tokens(self, probe, all_acts: torch.Tensor) -> torch.Tensor:
        """Compute probe scores for all tokens. Returns [num_tokens] tensor of scores."""
        if isinstance(all_acts, np.ndarray):
            all_acts = torch.from_numpy(all_acts)
        all_acts = all_acts.to(self.device)
        # all_acts is [num_tokens, hidden_dim]
        scores = probe(all_acts)  # [num_tokens] or [num_tokens, 1]
        if scores.dim() > 1:
            scores = scores.squeeze(-1)
        return scores

    @torch.no_grad()
    def evaluate_activations(
        self,
        activations_dict: Dict[int, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        env_name: str = "",
        trajectory_id: str = "",
        subenv_id: str = "",
        reward: float = 0.0,
        influence: float = 0.0,
    ) -> ProbeScores:
        """
        Evaluate activations using all probes at multiple positions.
        
        Args:
            activations_dict: {layer: {position: activations}} or {layer: activations}
                where position is 'first', 'last', 'avg', 'median', 'max', 'min', 'all'
                For first/last/avg/median: activations are [hidden_dim]
                For 'all': activations are [num_tokens, hidden_dim]
                max/min scores are computed from 'all' activations
        
        Returns:
            ProbeScores with all probe outputs at all positions
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
        
        # Determine if we have multi-position format or legacy single-tensor format
        sample_val = next(iter(activations_dict.values())) if activations_dict else None
        is_multi_position = isinstance(sample_val, dict)
        
        # Positions that use pre-computed activations
        PRECOMPUTED_POS = ['first', 'last', 'avg', 'median']
        
        # Truth probes
        for layer, probe in self.truth_probes.items():
            if layer not in activations_dict:
                continue
            
            layer_acts = activations_dict[layer]
            scores.truth_scores[layer] = {}
            
            if is_multi_position:
                # Compute scores for pre-computed positions
                for pos in PRECOMPUTED_POS:
                    if pos in layer_acts:
                        acts = layer_acts[pos]
                        if isinstance(acts, np.ndarray):
                            acts = torch.from_numpy(acts)
                        acts = acts.to(self.device)
                        score = probe(acts).mean().item()
                        scores.truth_scores[layer][pos] = score
                
                # Compute max/min from all token scores
                if 'all' in layer_acts:
                    all_scores = self._compute_scores_for_all_tokens(probe, layer_acts['all'])
                    scores.truth_scores[layer]['max'] = all_scores.max().item()
                    scores.truth_scores[layer]['min'] = all_scores.min().item()
            else:
                # Legacy format - single tensor, treat as 'last'
                acts = layer_acts
                if isinstance(acts, np.ndarray):
                    acts = torch.from_numpy(acts)
                acts = acts.to(self.device)
                score = probe(acts).mean().item()
                scores.truth_scores[layer]['last'] = score
        
        # Compute truth ensemble per position
        for pos in POSITIONS:
            pos_scores = [scores.truth_scores[l].get(pos) for l in scores.truth_scores if pos in scores.truth_scores.get(l, {})]
            pos_scores = [s for s in pos_scores if s is not None]
            if pos_scores:
                scores.truth_ensemble[pos] = float(np.mean(pos_scores))
        
        # Deception probes
        for layer, probe in self.deception_probes.items():
            if layer not in activations_dict:
                continue
            
            layer_acts = activations_dict[layer]
            scores.deception_scores[layer] = {}
            
            if is_multi_position:
                # Compute scores for pre-computed positions
                for pos in PRECOMPUTED_POS:
                    if pos in layer_acts:
                        acts = layer_acts[pos]
                        if isinstance(acts, np.ndarray):
                            acts = torch.from_numpy(acts)
                        acts = acts.to(self.device)
                        score = probe(acts).mean().item()
                        scores.deception_scores[layer][pos] = score
                
                # Compute max/min from all token scores
                if 'all' in layer_acts:
                    all_scores = self._compute_scores_for_all_tokens(probe, layer_acts['all'])
                    scores.deception_scores[layer]['max'] = all_scores.max().item()
                    scores.deception_scores[layer]['min'] = all_scores.min().item()
            else:
                acts = layer_acts
                if isinstance(acts, np.ndarray):
                    acts = torch.from_numpy(acts)
                acts = acts.to(self.device)
                score = probe(acts).mean().item()
                scores.deception_scores[layer]['last'] = score
        
        # Compute deception ensemble per position
        for pos in POSITIONS:
            pos_scores = [scores.deception_scores[l].get(pos) for l in scores.deception_scores if pos in scores.deception_scores.get(l, {})]
            pos_scores = [s for s in pos_scores if s is not None]
            if pos_scores:
                scores.deception_ensemble[pos] = float(np.mean(pos_scores))
        
        # Sycophancy probes
        for layer, probe in self.sycophancy_probes.items():
            if layer not in activations_dict:
                continue
            
            layer_acts = activations_dict[layer]
            scores.sycophancy_scores[layer] = {}
            
            if is_multi_position:
                # Compute scores for pre-computed positions
                for pos in PRECOMPUTED_POS:
                    if pos in layer_acts:
                        acts = layer_acts[pos]
                        if isinstance(acts, np.ndarray):
                            acts = torch.from_numpy(acts)
                        acts = acts.to(self.device)
                        score = probe(acts).mean().item()
                        scores.sycophancy_scores[layer][pos] = score
                
                # Compute max/min from all token scores
                if 'all' in layer_acts:
                    all_scores = self._compute_scores_for_all_tokens(probe, layer_acts['all'])
                    scores.sycophancy_scores[layer]['max'] = all_scores.max().item()
                    scores.sycophancy_scores[layer]['min'] = all_scores.min().item()
            else:
                acts = layer_acts
                if isinstance(acts, np.ndarray):
                    acts = torch.from_numpy(acts)
                acts = acts.to(self.device)
                score = probe(acts).mean().item()
                scores.sycophancy_scores[layer]['last'] = score
        
        # Compute sycophancy ensemble per position
        for pos in POSITIONS:
            pos_scores = [scores.sycophancy_scores[l].get(pos) for l in scores.sycophancy_scores if pos in scores.sycophancy_scores.get(l, {})]
            pos_scores = [s for s in pos_scores if s is not None]
            if pos_scores:
                scores.sycophancy_ensemble[pos] = float(np.mean(pos_scores))
        
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
        """Compute aggregated metrics for the iteration with multi-position support."""
        if not self.iteration_scores:
            return {}
        
        metrics = {}
        
        # For each position, compute aggregate metrics
        for pos in POSITIONS:
            # Truth metrics
            truth_vals = [s.truth_ensemble.get(pos) for s in self.iteration_scores if s.truth_ensemble.get(pos) is not None]
            if truth_vals:
                metrics[f"probe/truth_{pos}_mean"] = float(np.mean(truth_vals))
                metrics[f"probe/truth_{pos}_std"] = float(np.std(truth_vals))
            
            # Deception metrics
            deception_vals = [s.deception_ensemble.get(pos) for s in self.iteration_scores if s.deception_ensemble.get(pos) is not None]
            if deception_vals:
                metrics[f"probe/deception_{pos}_mean"] = float(np.mean(deception_vals))
                metrics[f"probe/deception_{pos}_std"] = float(np.std(deception_vals))
            
            # Sycophancy metrics
            sycophancy_vals = [s.sycophancy_ensemble.get(pos) for s in self.iteration_scores if s.sycophancy_ensemble.get(pos) is not None]
            if sycophancy_vals:
                metrics[f"probe/sycophancy_{pos}_mean"] = float(np.mean(sycophancy_vals))
                metrics[f"probe/sycophancy_{pos}_std"] = float(np.std(sycophancy_vals))
        
        # Per-layer per-position metrics
        for layer in self.truth_probes.keys():
            for pos in POSITIONS:
                vals = [s.truth_scores.get(layer, {}).get(pos) for s in self.iteration_scores]
                vals = [v for v in vals if v is not None]
                if vals:
                    metrics[f"probe/truth_L{layer}_{pos}_mean"] = float(np.mean(vals))
        
        for layer in self.deception_probes.keys():
            for pos in POSITIONS:
                vals = [s.deception_scores.get(layer, {}).get(pos) for s in self.iteration_scores]
                vals = [v for v in vals if v is not None]
                if vals:
                    metrics[f"probe/deception_L{layer}_{pos}_mean"] = float(np.mean(vals))
        
        for layer in self.sycophancy_probes.keys():
            for pos in POSITIONS:
                vals = [s.sycophancy_scores.get(layer, {}).get(pos) for s in self.iteration_scores]
                vals = [v for v in vals if v is not None]
                if vals:
                    metrics[f"probe/sycophancy_L{layer}_{pos}_mean"] = float(np.mean(vals))
        
        # Correlations with reward/influence (use 'avg' position as primary)
        rewards = np.array([s.reward for s in self.iteration_scores])
        influences = np.array([s.influence for s in self.iteration_scores])
        
        for pos in ['avg', 'last']:
            truth_ensemble = np.array([s.truth_ensemble.get(pos, 0) for s in self.iteration_scores])
            deception_ensemble = np.array([s.deception_ensemble.get(pos, 0) for s in self.iteration_scores])
            
            if len(rewards) > 1 and np.std(rewards) > 1e-6 and np.std(truth_ensemble) > 1e-6:
                metrics[f"probe/truth_{pos}_reward_corr"] = float(np.corrcoef(truth_ensemble, rewards)[0, 1])
            if len(rewards) > 1 and np.std(rewards) > 1e-6 and np.std(deception_ensemble) > 1e-6:
                metrics[f"probe/deception_{pos}_reward_corr"] = float(np.corrcoef(deception_ensemble, rewards)[0, 1])
            if len(influences) > 1 and np.std(influences) > 1e-6 and np.std(deception_ensemble) > 1e-6:
                metrics[f"probe/deception_{pos}_influence_corr"] = float(np.corrcoef(deception_ensemble, influences)[0, 1])
        
        return metrics
    
    def log_to_wandb(
        self,
        chosen_indices: Optional[List[int]] = None,
        rejected_indices: Optional[List[int]] = None,
        step: Optional[int] = None,
    ) -> Dict[str, float]:
        """Log iteration metrics to wandb with multi-position support."""
        metrics = self.compute_iteration_metrics(chosen_indices, rejected_indices)
        
        if metrics and WANDB_AVAILABLE:
            # Add histograms for each position
            for pos in POSITIONS:
                truth_vals = [s.truth_ensemble.get(pos) for s in self.iteration_scores if s.truth_ensemble.get(pos) is not None]
                deception_vals = [s.deception_ensemble.get(pos) for s in self.iteration_scores if s.deception_ensemble.get(pos) is not None]
                
                if truth_vals:
                    metrics[f"probe/truth_{pos}_histogram"] = wandb.Histogram(truth_vals)
                if deception_vals:
                    metrics[f"probe/deception_{pos}_histogram"] = wandb.Histogram(deception_vals)
            
            wandb.log(metrics, step=step)
        
        # Clear for next iteration
        self.clear_scores()
        
        return metrics
    
    def save_scores(self, path: str):
        """Save collected scores to JSON."""
        data = [s.to_dict() for s in self.iteration_scores]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


    def load_scores_from_trajectories(self, traj_df) -> int:
        """
        Load probe scores from trajectory dataframe.
        The traj_df should have a 'probe_scores' column containing score dicts.
        
        Returns:
            Number of scores loaded
        """
        self.clear_scores()
        count = 0
        
        if 'probe_scores' not in traj_df.columns:
            return 0
        
        for _, row in traj_df.iterrows():
            probe_scores_data = row.get('probe_scores')
            if probe_scores_data is None:
                continue
            
            # Handle both dict and string (JSON) formats
            if isinstance(probe_scores_data, str):
                try:
                    probe_scores_data = json.loads(probe_scores_data)
                except:
                    continue
            
            if not isinstance(probe_scores_data, dict):
                continue
            
            # Create ProbeScores object from the data
            scores = ProbeScores(
                truth_scores=probe_scores_data.get('truth_scores', {}),
                truth_ensemble=probe_scores_data.get('truth_ensemble', {}),
                deception_scores=probe_scores_data.get('deception_scores', {}),
                deception_ensemble=probe_scores_data.get('deception_ensemble', {}),
                sycophancy_scores=probe_scores_data.get('sycophancy_scores', {}),
                sycophancy_ensemble=probe_scores_data.get('sycophancy_ensemble', {}),
                env_name=probe_scores_data.get('env_name', ''),
                trajectory_id=probe_scores_data.get('trajectory_id', ''),
                subenv_id=probe_scores_data.get('subenv_id', ''),
                reward=probe_scores_data.get('reward', row.get('traj_rew', 0.0)),
                influence=probe_scores_data.get('influence', row.get('influence_score', 0.0)),
            )
            self.add_score(scores)
            count += 1
        
        return count
