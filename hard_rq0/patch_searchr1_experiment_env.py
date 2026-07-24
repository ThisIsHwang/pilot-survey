from __future__ import annotations

import argparse
from pathlib import Path

LEGACY_MARKER = "# STACKPILOT_EXPERIMENT_ENV_V1"
MARKER = "# STACKPILOT_EXPERIMENT_ENV_V2"
OLD_RAY_INIT = "        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})"
LEGACY_RAY_INIT = '''        # STACKPILOT_EXPERIMENT_ENV_V1
        ray_env = {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}
        for env_name in (
            'RQ0_SEED',
            'SEARCH_R1_MIXED_MODE',
            'SEARCH_R1_N_AGENT',
            'SEARCH_R1_RETRIEVER_TIMEOUT',
            'ANSWER_REWARD_WEIGHT',
            'EVIDENCE_REWARD_WEIGHT',
            'SEARCH_COST_WEIGHT',
            'PYTHONPATH',
        ):
            if env_name in os.environ:
                ray_env[env_name] = os.environ[env_name]
        ray.init(runtime_env={'env_vars': ray_env})'''
NEW_RAY_INIT = '''        # STACKPILOT_EXPERIMENT_ENV_V1
        # STACKPILOT_EXPERIMENT_ENV_V2
        ray_env = {
            'TOKENIZERS_PARALLELISM': 'true',
            'NCCL_DEBUG': os.environ.get('NCCL_DEBUG', 'WARN'),
        }
        for env_name in (
            'RQ0_SEED',
            'SEARCH_R1_MIXED_MODE',
            'SEARCH_R1_N_AGENT',
            'SEARCH_R1_RETRIEVER_TIMEOUT',
            'ANSWER_REWARD_WEIGHT',
            'EVIDENCE_REWARD_WEIGHT',
            'SEARCH_COST_WEIGHT',
            'PYTHONPATH',
            'PYTHONFAULTHANDLER',
            'TORCH_SHOW_CPP_STACKTRACES',
            'TORCH_DISTRIBUTED_DEBUG',
            'NCCL_DEBUG_SUBSYS',
        ):
            if env_name in os.environ:
                ray_env[env_name] = os.environ[env_name]
        ray_init_kwargs = {'runtime_env': {'env_vars': ray_env}}
        ray_temp_dir = os.environ.get('STACKPILOT_RAY_TMP_DIR')
        if ray_temp_dir:
            ray_init_kwargs['_temp_dir'] = ray_temp_dir
        ray.init(**ray_init_kwargs)'''


def patch(search_r1_root: Path) -> None:
    target = search_r1_root / "verl" / "trainer" / "main_ppo.py"
    text = target.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"Experiment-env patch already present: {target}")
        return
    if "import os\n" not in text:
        anchor = "import re\nimport numpy as np\n"
        if anchor not in text:
            raise RuntimeError(f"Pinned import anchor not found in {target}")
        text = text.replace(anchor, anchor + "import os\n", 1)
    if LEGACY_RAY_INIT in text:
        text = text.replace(LEGACY_RAY_INIT, NEW_RAY_INIT, 1)
    else:
        count = text.count(OLD_RAY_INIT)
        if count != 1:
            raise RuntimeError(
                f"Expected one pinned or legacy ray.init block in {target}; "
                f"found {count}"
            )
        text = text.replace(OLD_RAY_INIT, NEW_RAY_INIT, 1)
    target.write_text(text, encoding="utf-8")
    print(f"Applied numbered-experiment Ray env patch: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-r1-root", required=True)
    args = parser.parse_args()
    patch(Path(args.search_r1_root).resolve())
