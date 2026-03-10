from omegaconf import OmegaConf

cfg = OmegaConf.create({"ray_kwargs": {"ray_init": {}}})
OmegaConf.set_struct(cfg, True)

try:
    OmegaConf.update(cfg, "ray_kwargs.ray_init.runtime_env.env_vars.GRPO_COMPOSER_CONFIG", "my_json", force_add=True)
    print("SUCCESS")
    print(cfg.ray_kwargs.ray_init.runtime_env.env_vars.GRPO_COMPOSER_CONFIG)
except Exception as e:
    print(f"FAILED: {e}")
