from dataclasses import asdict, dataclass, fields, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

from targeted_llm_manipulation.config.accelerate_config import ACCELERATE_CONFIG_MAPPING, AccelerateConfigDeepSpeed
from targeted_llm_manipulation.root import EXPERIMENT_CONFIGS_DIR
from targeted_llm_manipulation.utils.utils import load_yaml

# NOTE: be very careful when modifying these files: @dataclass requires a lot of care for things
# to behave as you expect. Often you need to add a lot of type hints to make things work as expected.


T = TypeVar("T", bound="BaseExperimentConfig")


@dataclass
class BaseExperimentConfig:

    run_name: Optional[str] = None
    devices: List[int] = field(default_factory=list)

    # Env args
    env_class: str = ""
    env_fractions: Optional[Dict[str, float]] = None
    envs: Optional[List[str]] = None
    max_turns: int = 0
    num_envs_per_device: int = 1
    veto_prompt_type: str = ""

    subenv_choice_scheme: str = "sequential"
    pm_length_penalty: Optional[float] = None
    traj_selection_level: str = "subenv"

    # Baseiteration args
    n_subenvs_to_sample_per_env: int = 1  # Number of initial states to use for each iteration of training, per environment
    n_trajs_to_sample_per_subenv: int = 1  # Should generally be 1 unless traj_selection_level != subenv
    frac_selected_trajs: float = 0.5
    iterations: int = 1
    log_to_wandb: bool = True
    final_reward: bool = True

    # Veto args
    veto_level: Optional[float] = None
    allow_negative_training_on_veto: bool = False
    allow_id_to_see_tool_calls: bool = False

    # Training args
    model_names: Dict[str, str] = field(default_factory=dict)
    separate_agent_env_devices: str = ""
    inference_quantization: Optional[str] = None
    agent_max_tokens: Optional[int] = None  # Override agent max_tokens (useful for thinking mode models)
    
    # Probe evaluation args
    enable_probes: bool = False
    truth_probe_dir: Optional[str] = None
    deception_probe_dir: Optional[str] = None
    truth_probe_layers: Optional[List[int]] = None
    deception_probe_layers: Optional[List[int]] = None
    sycophancy_probe_dir: Optional[str] = None
    sycophancy_probe_layers: Optional[List[int]] = None

    # Debugging args
    seed: Optional[int] = None
    override_initial_traj_path: Optional[str] = None

    training_arg_keys = ["model_names"]

    # Static data for training (e.g. HH)
    static_dataset_name: Optional[str] = None
    frac_static_data_points: Optional[float] = None

    def __post_init__(self):
        # Convert frac_selected_trajs to a float if it's a string representing a fraction
        if isinstance(self.frac_selected_trajs, str):
            assert "/" in self.frac_selected_trajs, "Frac selected trajs should be a string of the form 'n/m'"
            terms = [float(x) for x in self.frac_selected_trajs.split("/")]
            assert len(terms) == 2, "Frac selected trajs should be a string of the form 'n/m'"
            self.frac_selected_trajs = terms[0] / terms[1]

    @classmethod
    def load(cls: Type[T], config_name: str, gpu_subset: Optional[List[int]] = None, verbose: bool = True) -> T:
        # Find the config file in the experiment_configs directory, searching for it in subdirectories
        matching_configs = list(EXPERIMENT_CONFIGS_DIR.rglob(config_name))
        assert len(matching_configs) > 0, f"No matching config file for {config_name}"
        assert len(matching_configs) < 2, f"More than one matching config file for {config_name}: {matching_configs}"

        config_path = matching_configs[0]
        config_dict = load_yaml(config_path)

        if "parent_config_to_override" in config_dict:
            # If there is a parent config defined, we should basically use that and only
            # override the keys that are in the current config
            parent_config_name = config_dict["parent_config_to_override"]
            print(f"To set up config {config_name}, going to first load parent config {parent_config_name}")
            parent_config = BaseExperimentConfig.load(parent_config_name, gpu_subset=gpu_subset, verbose=False)
            parent_config_dict = asdict(parent_config)
            del config_dict["parent_config_to_override"]
            print(f"Using params from {config_name} to override config params from {parent_config_name}")
            for k, v in config_dict.items():
                print(f"\tOverriding parameter {k}: \t{parent_config_dict.get(k, None)} → {v}")
            parent_config_dict.update(config_dict)
            config_dict = parent_config_dict

        if gpu_subset is not None:
            if verbose:
                print(f"GPU indices to run on: {gpu_subset}")
            config_dict["devices"] = gpu_subset
        else:
            import torch

            visible_devices = list(range(torch.cuda.device_count()))
            if verbose:
                print(f"Using all available CUDA devices {visible_devices}")
            config_dict["devices"] = visible_devices

        if verbose:
            print(f"Creating config from file {config_name}")

        # Set default values
        config_dict.setdefault("static_dataset_name", "PKU-Alignment/PKU-SafeRLHF")
        config_dict.setdefault("frac_static_data_points", 0)
        config_dict.setdefault("agent_max_tokens", None)  # Default: use env config value
        config_dict.setdefault("enable_probes", False)
        config_dict.setdefault("truth_probe_dir", "/home/user/truth_probe/Truth_is_Universal/checkpoints")
        config_dict.setdefault("deception_probe_dir", "/home/user/deception_probe/outputs/gemma2b_deception_probes/probes/pytorch/best")
        config_dict.setdefault("truth_probe_layers", [12, 14, 18])
        config_dict.setdefault("deception_probe_layers", [13, 14, 15])
        config_dict.setdefault("sycophancy_probe_dir", "/home/user/sycophancy_probe/probes_pt")
        config_dict.setdefault("sycophancy_probe_layers", [11, 12, 13])

        return cls.create_config(config_dict)

    @classmethod
    def create_config(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        if "beta" in config_dict:
            print("Creating KTO config")
            config_class = KTOConfig
        elif "n_train_epochs" in config_dict:
            print("Creating OpenAI Expert Iteration config")
            config_class = OpenAIExpertIterationConfig
        else:
            print("Creating Expert Iteration config")
            config_class = ExpertIterationConfig

        config_class._validate_config_keys(config_dict)

        return config_class(**config_dict)  # type: ignore

    @classmethod
    def _validate_config_keys(cls: Type[T], config_dict: Dict[str, Any]):
        # Get the set of all field names from the Config class
        all_fields = set(field.name for field in fields(cls))  # type: ignore

        # Check for any extra keys in the loaded config
        extra_keys = set(config_dict.keys()) - all_fields
        if extra_keys:
            raise ValueError(f"Unexpected configuration parameters: {', '.join(extra_keys)}")

        # Check for any missing keys in the loaded config (apart from a couple)
        missing_keys = all_fields - set(config_dict.keys())
        missing_keys = missing_keys - {"accelerate_config", "default_config_path"}
        if missing_keys:
            raise ValueError(
                f"Missing configuration parameters for {config_dict['run_name']}: {', '.join(missing_keys)}"
            )

        # Validating that individual attributes make sense
        if config_dict["override_initial_traj_path"] is not None:
            path = Path(config_dict["override_initial_traj_path"])
            # Check that the path is a jsonl file
            if not path.suffix == ".jsonl":
                raise ValueError(f"Override initial traj path should be a selected trajectories jsonl file: {path}")

        if config_dict["subenv_choice_scheme"] not in ["sequential", "random", "fixed"]:
            raise ValueError(
                f"Subenv choice scheme should be either 'sequential', 'random', or 'fixed': {config_dict['subenv_choice_scheme']}"
            )

        assert sum(config_dict["env_fractions"].values()) == 1, "Env fractions should sum to 1"
        # if config_dict["traj_selection_level"] != "subenv":
        #     assert (
        #         config_dict["n_trajs_to_sample_per_subenv"] == 1
        #     ), "Num gen trajs per subenv should be 1 unless traj_selection_level == subenv"

    @property
    def env_args(self):
        return {
            "env_class": self.env_class,
            "envs": self.envs,
            "max_turns": self.max_turns,
            "num_envs_per_device": self.num_envs_per_device,
            "n_subenvs_to_sample_per_env": self.n_subenvs_to_sample_per_env,
            "n_trajs_to_sample_per_subenv": self.n_trajs_to_sample_per_subenv,
            "subenv_choice_scheme": self.subenv_choice_scheme,
            "env_fractions": self.env_fractions,
            "allow_id_to_see_tool_calls": self.allow_id_to_see_tool_calls,
            "veto_prompt_type": self.veto_prompt_type,
        }

    @property
    def training_args(self):
        return {k: v for k, v in asdict(self).items() if k in self.training_arg_keys}


@dataclass
class LocalTrainingConfig(BaseExperimentConfig):

    # Training args - all with defaults to support dataclass inheritance
    per_device_train_batch_size: int = 1
    num_train_epochs: int = 1
    gradient_checkpointing: bool = True
    learning_rate: float = 1e-5
    report_to: str = "wandb"
    optim: str = "adamw_torch"
    max_length: int = 2048
    lr_scheduler_type: str = "cosine"  # (Within each iteration)
    across_iter_lr_mult_factor: float = 1.0  # (Across iterations) E.g. if 1/3, LR will be 1/3 of prev val after every iter
    logging_steps: int = 1
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_grad_norm: float = 1.0

    accelerate_config_type: str = "DeepSpeedZeRO2"
    effective_batch_size: int = 8

    def __post_init__(self):
        super().__post_init__()
        # NOTE: These shouldn't be necessary for most local training, but if one is doing some weird mixed training (GPT veto model), necessary to have these set
        self.max_tokens_per_minute = 500_000
        self.max_requests_per_minute = 5_000
        self.accelerate_config = ACCELERATE_CONFIG_MAPPING[self.accelerate_config_type]()
        print(f"Using {self.accelerate_config_type} Accelerate config")

        if "DeepSpeed" in self.accelerate_config_type:
            self.accelerate_config.set_gradient_clipping(self.max_grad_norm)  # type: ignore
            assert isinstance(self.accelerate_config, AccelerateConfigDeepSpeed)
            self.accelerate_config.set_gradient_clipping(self.max_grad_norm)

        self.training_arg_keys = self.training_arg_keys + [
            "per_device_train_batch_size",
            "num_train_epochs",
            "gradient_checkpointing",
            "learning_rate",
            "report_to",
            "optim",
            "max_length",
            "lr_scheduler_type",
            "across_iter_lr_mult_factor",
            "logging_steps",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "max_grad_norm",
        ]
        self.accelerate_config.set_gpu_ids(self.devices)
        self.accelerate_config.update_gradient_accumulation_steps(
            self.effective_batch_size, self.per_device_train_batch_size
        )

    @classmethod
    def _validate_config_keys(cls: Type[T], config_dict: Dict[str, Any]):
        super()._validate_config_keys(config_dict)


@dataclass
class ExpertIterationConfig(LocalTrainingConfig):
    pass


@dataclass
class OpenAIExpertIterationConfig(BaseExperimentConfig):

    batch_size: int = 1
    n_train_epochs: int = 1
    learning_rate_multiplier: float = 1.0

    max_tokens_per_minute: int = 500000
    max_requests_per_minute: int = 5000

    def __post_init__(self):
        super().__post_init__()
        self.training_arg_keys = self.training_arg_keys + [
            "batch_size",
            "n_train_epochs",
            "learning_rate_multiplier",
        ]


@dataclass
class KTOConfig(LocalTrainingConfig):

    beta: float = 0.1
    target_ratio: float = 0.5
    max_prompt_length: int = 1024
    max_completion_length: int = 1024

    def __post_init__(self):
        super().__post_init__()
        self.training_arg_keys = self.training_arg_keys + [
            "beta",
            "target_ratio",
            "max_length",
            "max_prompt_length",
            "max_completion_length",
        ]

    @classmethod
    def _validate_config_keys(cls: Type[T], config_dict: Dict[str, Any]):
        super()._validate_config_keys(config_dict)
        assert config_dict["max_prompt_length"] + config_dict["max_completion_length"] <= config_dict["max_length"]
