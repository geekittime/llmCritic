"""
Borrowed from verl.trainer.main_ppo.py
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from ragen.trainer.agent_trainer import RayAgentTrainer

import ray
import hydra
import os
from verl import DataProto
import torch
import numpy as np
from ragen.utils import register_resolvers
register_resolvers()
import sys
import socket
import stat
import warnings
from omegaconf import OmegaConf
from ragen.utils import redact_config


# Keep the private name used by the existing launcher/tests while sharing the
# implementation with the trainer's W&B configuration sanitization.
_redact_config = redact_config


_ALLOWED_CREDENTIAL_NAMES = {"DEEPSEEK_API_KEY", "WANDB_API_KEY"}


def _validate_credential_file(path_value):
    """Validate a credential file without reading secret values in the driver."""
    if path_value is None or not str(path_value).strip():
        return None
    path = os.path.abspath(os.path.expanduser(str(path_value)))
    file_stat = os.stat(path)
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"Credential path is not a regular file: {path}")
    if file_stat.st_uid != os.getuid():
        raise PermissionError(f"Credential file must be owned by uid {os.getuid()}: {path}")
    if stat.S_IMODE(file_stat.st_mode) & 0o077:
        raise PermissionError(f"Credential file must have mode 600 or stricter: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"Credential file is not readable: {path}")
    return path


def _load_task_credentials(path_value):
    """Load an allowlisted mode-600 file inside the Ray TaskRunner only."""
    path = _validate_credential_file(path_value)
    if path is None:
        return
    parsed = {}
    with open(path, "r", encoding="utf-8") as credential_file:
        for line_number, raw_line in enumerate(credential_file, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].lstrip()
            name, separator, value = line.partition("=")
            name = name.strip()
            if not separator or name not in _ALLOWED_CREDENTIAL_NAMES:
                raise ValueError(f"Unsupported credential assignment at {path}:{line_number}")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if not value:
                raise ValueError(f"Empty credential value at {path}:{line_number}")
            parsed[name] = value
    if "DEEPSEEK_API_KEY" not in parsed and not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        raise ValueError(f"Credential file does not define DEEPSEEK_API_KEY: {path}")
    for name, value in parsed.items():
        # An explicitly selected protected file must override stale inherited
        # credentials, especially after key rotation.
        os.environ[name] = value


def _validate_rollout_action_budgets(config) -> None:
    """Reject prompts that advertise more actions than rollout can execute."""
    max_model_len = OmegaConf.select(
        config, "actor_rollout_ref.rollout.max_model_len", default=None
    )
    response_length = OmegaConf.select(
        config, "actor_rollout_ref.rollout.response_length", default=None
    )
    if max_model_len is not None and response_length is not None:
        max_model_len = int(max_model_len)
        response_length = int(response_length)
        if response_length <= 0 or max_model_len <= response_length:
            raise ValueError(
                "actor_rollout_ref.rollout.max_model_len must be greater than the positive "
                "response_length so generation has a non-empty prompt budget"
            )

    max_turn = int(OmegaConf.select(config, "agent_proxy.max_turn", default=0) or 0)
    max_actions_per_turn = int(
        OmegaConf.select(config, "agent_proxy.max_actions_per_turn", default=0) or 0
    )
    if max_turn <= 0 or max_actions_per_turn <= 0:
        raise ValueError(
            "agent_proxy.max_turn and max_actions_per_turn must both be positive integers"
        )
    rollout_capacity = max_turn * max_actions_per_turn

    custom_envs = OmegaConf.select(config, "custom_envs", default=None)
    es_manager = OmegaConf.select(config, "es_manager", default=None)
    if custom_envs is None or es_manager is None:
        return

    checked: set[str] = set()
    for mode in ("train", "val"):
        mode_config = OmegaConf.select(config, f"es_manager.{mode}", default=None)
        if mode_config is None:
            continue
        tags = OmegaConf.select(mode_config, "env_configs.tags", default=[]) or []
        for raw_tag in tags:
            tag = str(raw_tag)
            if tag in checked:
                continue
            checked.add(tag)
            env_config = custom_envs.get(tag)
            if env_config is None:
                raise ValueError(f"es_manager.{mode} selects unknown custom_env tag {tag!r}")
            env_budget = int(env_config.get("max_actions_per_traj", 0) or 0)
            if env_budget <= 0:
                raise ValueError(
                    f"custom_envs.{tag}.max_actions_per_traj must be a positive integer"
                )
            if env_budget > rollout_capacity:
                raise ValueError(
                    f"custom_envs.{tag}.max_actions_per_traj={env_budget} exceeds rollout "
                    f"capacity agent_proxy.max_turn * max_actions_per_turn={rollout_capacity}; "
                    "the model would be promised actions that cannot be executed"
                )

class DummyRewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {
                    "reward_tensor": data.batch['rm_scores'],
                }
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        all_scores = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            score = data_item.non_tensor_batch['reward']
            score = float(score)
 
            reward_tensor[i, valid_response_length - 1] = score
            all_scores.append(score)

            # Get data_source from data_item if available, otherwise use a default value
            data_source = data_item.non_tensor_batch.get('data_source', 'default')
            
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
        
        print(f"[DEBUG] all_scores: {all_scores}")
        print(f"[DEBUG] all_scores shape: {np.array(all_scores).shape}")
        print(f"[DEBUG] all_scores mean: {np.mean(all_scores)}")
        print(f"[DEBUG] all_scores max: {np.max(all_scores)}")
        print(f"[DEBUG] all_scores min: {np.min(all_scores)}")
        print(f"[DEBUG] all_scores std: {np.std(all_scores)}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
            }
        else:
            return reward_tensor

def get_custom_reward_fn(config):
    import importlib.util, os

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    if spec is None:
        raise RuntimeError(f"Failed to create module spec from '{file_path}'")
        
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")
    if not function_name:
        raise ValueError("Function name not specified in custom_reward_function config")

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")

    return getattr(module, function_name)



def add_dependency_and_validate_config(config):

    # validate config
    _validate_rollout_action_budgets(config)
    assert config.micro_batch_size_per_gpu * config.trainer.n_gpus_per_node <= config.actor_rollout_ref.actor.ppo_mini_batch_size, \
        f"micro_batch_size_per_gpu * n_gpus_per_node ({config.micro_batch_size_per_gpu * config.trainer.n_gpus_per_node}) must be less than or equal to ppo_mini_batch_size ({config.actor_rollout_ref.actor.ppo_mini_batch_size})"
    assert config.actor_rollout_ref.actor.ppo_mini_batch_size % (config.micro_batch_size_per_gpu * config.trainer.n_gpus_per_node) == 0, \
        f"ppo_mini_batch_size ({config.actor_rollout_ref.actor.ppo_mini_batch_size}) must be divisible by micro_batch_size_per_gpu * n_gpus_per_node ({config.micro_batch_size_per_gpu * config.trainer.n_gpus_per_node})"
    assert "qwen" in config.model_path.lower() or (not config.enable_response_mask), \
        "response mask is currently only supported for qwen models"
    assert len(str(config.system.CUDA_VISIBLE_DEVICES).split(',')) == config.trainer.n_gpus_per_node, \
        f"CUDA_VISIBLE_DEVICES ({config.system.CUDA_VISIBLE_DEVICES}) must have the same number of GPUs as n_gpus_per_node ({config.trainer.n_gpus_per_node})"
    context_window_mode = getattr(config.agent_proxy, "context_window_mode", "full")
    if context_window_mode in ("single_turn", "limited_multi_turn"):
        # In these modes, each turn becomes a separate sample, so we need more samples
        assert config.es_manager.train.env_groups * config.es_manager.train.group_size * config.actor_rollout_ref.rollout.rollout_filter_ratio * config.agent_proxy.max_turn >= config.ppo_mini_batch_size, \
            f"env_groups * group_size * rollout_filter_ratio * max_turn ({config.es_manager.train.env_groups * config.es_manager.train.group_size * config.actor_rollout_ref.rollout.rollout_filter_ratio * config.agent_proxy.max_turn}) must be greater than or equal to ppo_mini_batch_size ({config.ppo_mini_batch_size})"
    else:
        assert config.es_manager.train.env_groups * config.es_manager.train.group_size * config.actor_rollout_ref.rollout.rollout_filter_ratio >= config.ppo_mini_batch_size, \
            f"env_groups * group_size * rollout_filter_ratio ({config.es_manager.train.env_groups * config.es_manager.train.group_size * config.actor_rollout_ref.rollout.rollout_filter_ratio}) must be greater than or equal to ppo_mini_batch_size ({config.ppo_mini_batch_size}). Note that effective rollouts for update is env_groups * group_size * rollout_filter_ratio."
    assert config.algorithm.bi_level_gae == False or config.algorithm.adv_estimator == "gae", "BI_LEVEL_GAE is enabled, so config.algorithm.adv_estimator should be set to gae"
    assert config.algorithm.bi_level_gae == False or (not config.agent_proxy.use_turn_scores), "BI_LEVEL_GAE is enabled, but currently use_turn_scores are not correctly supported, so config.agent_proxy.use_turn_scores should be set to False" # This will be added later. Currently turn-scores are not correctly supported yet.
    # assert config.algorithm.bi_level_gae == False or config.agent_proxy.use_turn_scores, "BI_LEVEL_GAE is enabled, so config.agent_proxy.use_turn_scores should be set to True" # This will be added later. Currently turn-scores are not correctly supported yet.

    # The exact turn-PPO path relies on the original rollout token traces.
    # Fail at launch time instead of silently falling back to a different
    # probability model or producing an unaligned loss mask.
    use_turn_ppo = bool(config.algorithm.get("use_label_outcome_advantage", False))
    if use_turn_ppo and context_window_mode != "full":
        raise ValueError(
            "algorithm.use_label_outcome_advantage requires agent_proxy.context_window_mode=full "
            "so each sampled turn can retain exact prompt/response token IDs"
        )
    if use_turn_ppo:
        if int(config.actor_rollout_ref.rollout.n) != 1:
            raise ValueError(
                "Exact turn PPO requires actor_rollout_ref.rollout.n=1; environment "
                "group_size already controls repeated trajectories"
            )
        turn_advantage_mode = str(config.algorithm.get("turn_advantage_mode", "weighted"))
        if turn_advantage_mode not in {"weighted", "label_only"}:
            raise ValueError(
                "algorithm.turn_advantage_mode must be 'weighted' or 'label_only', "
                f"got {turn_advantage_mode!r}"
            )
        if turn_advantage_mode == "label_only":
            if not bool(config.get("generative_critic", {}).get("enable", False)):
                raise ValueError(
                    "label_only requires generative_critic.enable=True so the turn "
                    "advantage comes from a judge"
                )
            if str(config.algorithm.get("turn_score_reduction", "mean")) != "mean":
                raise ValueError(
                    "label_only requires algorithm.turn_score_reduction=mean so each "
                    "turn keeps the judge's exact {-1, 0, 1} score"
                )
            if bool(config.algorithm.get("normalize_turn_advantage", False)):
                raise ValueError(
                    "label_only requires algorithm.normalize_turn_advantage=False so the "
                    "optimized advantage remains exactly the judge output"
                )
            if bool(config.algorithm.use_kl_in_reward) and bool(
                config.algorithm.get("add_kl_to_turn_advantage", True)
            ):
                raise ValueError(
                    "label_only is incompatible with a KL reward contribution; set "
                    "algorithm.use_kl_in_reward=False"
                )
        turn_credit_assignment = str(
            config.algorithm.get("turn_credit_assignment", "direct")
        )
        if turn_credit_assignment not in {"direct", "discounted_return"}:
            raise ValueError(
                "algorithm.turn_credit_assignment must be 'direct' or "
                f"'discounted_return', got {turn_credit_assignment!r}"
            )
        if turn_credit_assignment == "discounted_return":
            if (
                turn_advantage_mode != "label_only"
                and float(config.algorithm.get("outcome_weight", 1.0)) != 0.0
                and str(config.algorithm.get("outcome_broadcast", "all_turns")) == "all_turns"
            ):
                raise ValueError(
                    "discounted_return cannot accumulate an outcome broadcast to all turns; "
                    "use label_only or outcome_broadcast=last_turn/none"
                )
            warnings.warn(
                "turn_credit_assignment=discounted_return treats turn scores as rewards and "
                "uses episode return-to-go. It is not GAE: no learned value baseline is used.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            direct_score_description = (
                "each judge label"
                if turn_advantage_mode == "label_only"
                else "each composed judge/outcome turn score"
            )
            warnings.warn(
                f"turn_credit_assignment=direct uses {direct_score_description} directly as the policy "
                "advantage. Neutral labels produce zero actor gradient; despite "
                "algorithm.adv_estimator=gae, this path is not GAE and has no value baseline.",
                RuntimeWarning,
                stacklevel=2,
            )
        critic_enable = config.critic.get("enable", None)
        value_critic_enabled = bool(critic_enable) or (
            critic_enable is None and config.algorithm.adv_estimator == "gae"
        )
        if value_critic_enabled:
            raise ValueError(
                "Exact turn PPO currently requires critic.enable=False: expanded one-turn rows "
                "cannot run cross-row value GAE without regrouping episode_ids"
            )
        if int(config.actor_rollout_ref.actor.ulysses_sequence_parallel_size) != 1:
            raise ValueError("Exact turn PPO currently requires ulysses_sequence_parallel_size=1")
        if float(config.actor_rollout_ref.actor.entropy_coeff) != 0.0:
            raise ValueError("Exact turn PPO currently requires actor entropy_coeff=0")
        if bool(config.actor_rollout_ref.actor.use_kl_loss):
            raise ValueError(
                "Exact turn PPO does not support the token-level actor KL loss; "
                "use algorithm.use_kl_in_reward with a reference policy instead"
            )

    critic_enabled = bool(config.get("generative_critic", {}).get("enable", False))
    critic_backend = str(config.get("generative_critic", {}).get("backend", "transformers")).lower()
    critic_config = config.get("generative_critic", {})
    if critic_backend in {"deepseek", "deepseek_api"}:
        parse_fail_score = critic_config.get("parse_fail_score", 0)
        if parse_fail_score is not None and int(parse_fail_score) != 0:
            raise ValueError(
                "DeepSeek parse/API failures must map to neutral 0; set "
                "generative_critic.parse_fail_score=0"
            )

    audit_enabled = bool(critic_config.get("audit_enable", False))
    if audit_enabled:
        audit_path = str(critic_config.get("audit_path", "") or "").strip()
        audit_sample_rate = float(critic_config.get("audit_sample_rate", 0.1))
        audit_max_records = int(critic_config.get("audit_max_records", 5000))
        audit_raw_output_max_chars = int(
            critic_config.get("audit_raw_output_max_chars", 2000)
        )
        if not audit_path:
            raise ValueError("generative_critic.audit_path is required when audit_enable=True")
        if not 0.0 <= audit_sample_rate <= 1.0:
            raise ValueError("generative_critic.audit_sample_rate must be in [0, 1]")
        if audit_max_records < 0:
            raise ValueError("generative_critic.audit_max_records must be non-negative")
        if audit_raw_output_max_chars <= 0:
            raise ValueError(
                "generative_critic.audit_raw_output_max_chars must be positive"
            )
        credential_path = critic_config.get("deepseek_api_key_file", None)
        if credential_path and os.path.abspath(os.path.expanduser(str(credential_path))) == os.path.abspath(
            os.path.expanduser(audit_path)
        ):
            raise ValueError("critic audit path must not overwrite the credential file")
    if use_turn_ppo and critic_enabled and critic_backend in {"deepseek", "deepseek_api"}:
        key_env = str(
            config.get("generative_critic", {}).get("deepseek_api_key_env", "DEEPSEEK_API_KEY")
        )
        configured_key = config.get("generative_critic", {}).get("deepseek_api_key", None)
        credential_file = _validate_credential_file(
            config.get("generative_critic", {}).get("deepseek_api_key_file", None)
        )
        if (
            not os.environ.get(key_env, "").strip()
            and not str(configured_key or "").strip()
            and credential_file is None
        ):
            raise RuntimeError(
                f"DeepSeek critic is enabled but {key_env} is not set. "
                "Use a protected deepseek_api_key_file or export the key; "
                "do not put the value in Hydra overrides."
            )

    # add dependency
    config.data.train_batch_size = config.es_manager.train.env_groups * config.es_manager.train.group_size


    return config


@hydra.main(version_base=None, config_path="config", config_name="base")
def main(config):
    config = add_dependency_and_validate_config(config)
    print(f"config: {_redact_config(config)}")

    run_ppo(config)


def run_ppo(config) -> None:
    # TODO(linjunrong.ocss884): this ENV is left for resolving SGLang conflict with ray devices
    # isolation, will solve in the future
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.system.CUDA_VISIBLE_DEVICES)
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not ray.is_initialized():
        # this is for local ray cluster
        # Never put credentials in runtime_env: Ray exposes it through job
        # metadata and writes it to runtime-env logs. The reviewed launcher
        # supplies only a protected file path, read inside TaskRunner.run().
        ray_env_vars = {
            'TOKENIZERS_PARALLELISM': 'true',
            'NCCL_DEBUG': 'WARN',
            'VLLM_LOGGING_LEVEL': 'WARN',
            "RAY_DEBUG": "legacy",  # used here for simpler breakpoint()
        }
        ray_init_config = config.get("ray_kwargs", {}).get("ray_init", {})
        configured_num_cpus = ray_init_config.get("num_cpus", None)
        ray_init_kwargs = {
            "address": "local",
            "include_dashboard": False,
            "runtime_env": {"env_vars": ray_env_vars},
        }
        if configured_num_cpus is not None:
            ray_init_kwargs["num_cpus"] = int(configured_num_cpus)
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**ray_init_kwargs)

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:

    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        credential_file = config.get("generative_critic", {}).get(
            "deepseek_api_key_file", None
        )
        _load_task_credentials(credential_file)

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(_redact_config(OmegaConf.to_container(config, resolve=True)))
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_tokenizer, hf_processor
        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from ragen.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker, GenerativeCriticWorker
            from verl.single_controller.ray import RayWorkerGroup
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        use_trainable_generative_critic = bool(
            config.get("generative_critic", {}).get("train_enable", False)
        )
        critic_worker_cls = GenerativeCriticWorker if use_trainable_generative_critic else CriticWorker

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(critic_worker_cls),
        }
        if config.actor_rollout_ref.actor.use_ref:
            print("[DEBUG] using ref policy")
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        else:
            print("[DEBUG] not using ref policy, setting use_kl_loss to False")
            config.actor_rollout_ref.actor.use_kl_loss = False
        global_pool_id = 'global_pool'
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }
        if config.actor_rollout_ref.actor.use_ref:
            mapping[Role.RefPolicy] = global_pool_id
        # mapping = {
        #     Role.ActorRollout: global_pool_id,
        #     Role.Critic: global_pool_id,
        #     Role.RefPolicy: global_pool_id,
        # }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy == 'fsdp':
                from ragen.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == 'megatron':
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # reward_manager_name = config.reward_model.get("reward_manager", "dummy")
        # print(f'reward_manager_name: {reward_manager_name}')
        # if reward_manager_name == 'dummy':
        print("using dummy reward manager")
        reward_manager_cls = DummyRewardManager
        # elif reward_manager_name == 'naive':
        #     from verl.workers.reward_manager import NaiveRewardManager
        #     reward_manager_cls = NaiveRewardManager
        # elif reward_manager_name == 'prime':
        #     from verl.workers.reward_manager import PrimeRewardManager
        #     reward_manager_cls = PrimeRewardManager
        # else:
        #     raise NotImplementedError

        compute_score = get_custom_reward_fn(config)
        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)

        # Note that we always use function-based RM for validation
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = RayAgentTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn
        )
        trainer.init_workers()
        trainer.init_agent_proxy()
        trainer.fit()


if __name__ == '__main__':
    import sys
    sys.argv.extend([
        "--config-dir", os.path.join(os.path.dirname(__file__), "ragen/config"),
        "--config-dir", os.path.join(os.path.dirname(__file__), "verl/verl/trainer/config"),
    ])
    main()
