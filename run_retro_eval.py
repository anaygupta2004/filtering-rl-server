import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import sys
sys.path.insert(0, '/home/user/manipulation-llm')

from targeted_llm_manipulation.retroactive_evaluator.run_retroactive_evals import evaluate_runs_hf
from targeted_llm_manipulation.utils.utils import find_freest_gpus

if __name__ == '__main__':
    run = "gemma_2b_scratchpad_prefill-12-30_16-30-00"
    metrics = ["omission", "convincing_not_to_book", "implies_booking_succeeded", "error_mentioning"]

    backend_config = {
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "lora_path": None,
    }

    devices = find_freest_gpus(2)
    print(f"Using devices: {devices}")

    evaluate_runs_hf(
        runs=[run],
        backend_config=backend_config,
        devices=devices,
        batch_size=8,
        max_trajs_per_env=20,
        iterations_list=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
        metrics_list=[metrics],
        env_config_name=None,
        training_run=True,
        benchmark=False,
    )
