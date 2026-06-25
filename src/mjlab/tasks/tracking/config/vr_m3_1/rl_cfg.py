"""RL configuration for VR M3.1 tracking task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def vr_m3_1_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for VR M3.1 tracking task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(1024, 512, 256),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(1024, 512, 256),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="vr_m3_1_tracking",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=100_000,
  )


def vr_m3_1_pose_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the pose-only VR M3.1 runner configuration."""
  cfg = vr_m3_1_tracking_ppo_runner_cfg()
  cfg.obs_groups = {
    "actor": ("tokenizer", "actor_g1", "actor_teleop", "actor_smpl", "proprioception"),
    "critic": ("critic",),
    "tokenizer": ("tokenizer",),
    "actor_g1": ("actor_g1",),
    "actor_teleop": ("actor_teleop",),
    "actor_smpl": ("actor_smpl",),
    "proprioception": ("proprioception",),
  }
  cfg.actor.class_name = "mjlab.tasks.tracking.rl.universal:UniversalTokenActor"
  cfg.algorithm.class_name = "mjlab.tasks.tracking.rl.universal:UniversalTokenPPO"
  cfg.actor.hidden_dims = (512, 256, 128)
  cfg.actor.num_fsq_levels = 32
  cfg.actor.fsq_level_list = 32
  cfg.actor.max_num_tokens = 2
  cfg.actor.encoder_names = ("g1", "teleop", "smpl")
  cfg.critic.hidden_dims = (512, 256, 128)
  cfg.experiment_name = "vr_m3_1_pose"
  cfg.run_name = "from_scratch"
  cfg.max_iterations = 150_000
  cfg.save_interval = 1000
  cfg.algorithm.learning_rate = 5.0e-4
  cfg.algorithm.entropy_coef = 0.008
  cfg.algorithm.desired_kl = 0.01
  cfg.algorithm.aux_loss_coefs = {
    "g1_recon": 0.01,
    "g1_smpl_latent": 1.0,
    "g1_teleop_latent": 1.0,
    "teleop_smpl_latent": 1.0,
    "reencoded_smpl_g1_latent": 1.0,
  }
  # Healthy latent-distillation losses sit ~1-2.5; cap each term's contribution
  # at 8 so a divergence spike (observed ~220 when a batch of hard motions
  # entered the curriculum) cannot dominate the gradient and blow up the shared
  # trunk. Leaves ~3x headroom over normal operation; see UniversalTokenPPO.
  cfg.algorithm.aux_loss_cap = 8.0
  return cfg


def vr_m3_1_teleop_robust_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the from-scratch teleop-robust VR M3.1 runner configuration."""
  cfg = vr_m3_1_tracking_ppo_runner_cfg()
  cfg.experiment_name = "vr_m3_1_teleop_robust"
  cfg.run_name = "from_scratch"
  cfg.max_iterations = 300_000
  cfg.save_interval = 1000
  cfg.algorithm.learning_rate = 5.0e-4
  cfg.algorithm.entropy_coef = 0.01
  return cfg


def vr_m3_1_balance_finetune_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = vr_m3_1_tracking_ppo_runner_cfg()
  cfg.experiment_name = "vr_m3_1_balance_finetune"
  cfg.run_name = "from_tracking_checkpoint"
  cfg.max_iterations = 20_000
  cfg.save_interval = 500
  cfg.algorithm.learning_rate = 5.0e-5
  cfg.algorithm.entropy_coef = 0.0
  cfg.algorithm.desired_kl = 0.003
  cfg.algorithm.clip_param = 0.10
  cfg.algorithm.num_learning_epochs = 3
  cfg.algorithm.max_grad_norm = 0.5
  return cfg


def vr_m3_1_balance_transition_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = vr_m3_1_tracking_ppo_runner_cfg()
  cfg.experiment_name = "vr_m3_1_balance_transition"
  cfg.run_name = "from_scratch"
  cfg.max_iterations = 120_000
  cfg.save_interval = 1000
  cfg.algorithm.learning_rate = 3.0e-4
  cfg.algorithm.entropy_coef = 0.005
  cfg.algorithm.desired_kl = 0.008
  cfg.algorithm.clip_param = 0.15
  cfg.algorithm.num_learning_epochs = 4
  cfg.algorithm.max_grad_norm = 0.8
  return cfg
