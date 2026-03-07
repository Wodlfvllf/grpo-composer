"""
Patch veRL's core_algos.py to auto-register the grpo_composer "composer" loss.

Problem: @register_policy_loss("composer") only runs in the main process
(train_grpo.py). Ray worker processes call get_policy_loss_fn("composer")
but never imported grpo_composer, so "composer" isn't in the registry.

Fix: Patch get_policy_loss_fn() to lazily import grpo_composer losses
on first call (avoiding circular import issues).

Run during Docker/Modal image build AFTER pip installing veRL + grpo_composer.
"""

import importlib.util


def patch():
    spec = importlib.util.find_spec("verl.trainer.ppo.core_algos")
    if spec is None or spec.origin is None:
        print("verl core_algos not found, skipping patch")
        return

    filepath = spec.origin
    with open(filepath, "r") as f:
        source = f.read()

    # Check if already patched
    if "grpo_composer" in source:
        print(f"Already patched: {filepath}")
        return

    # We patch get_policy_loss_fn to lazily load our module on first call.
    # This avoids circular imports (our module imports from this same file).
    old = "def get_policy_loss_fn(name):"
    new = (
        "def get_policy_loss_fn(name):\n"
        "    # Auto-register grpo_composer losses on first lookup (patched)\n"
        "    if 'composer' not in POLICY_LOSS_REGISTRY:\n"
        "        try:\n"
        "            import grpo_composer.integrations.verl.losses  # noqa: F401\n"
        "        except Exception:\n"
        "            pass\n"
    )

    if old not in source:
        print(f"Could not find '{old}' in {filepath}")
        return

    source = source.replace(old, new, 1)

    with open(filepath, "w") as f:
        f.write(source)

    print(f"Successfully patched get_policy_loss_fn in: {filepath}")


if __name__ == "__main__":
    patch()
