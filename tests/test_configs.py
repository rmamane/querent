"""Every arm + experiment config composes through Hydra and instantiates a model.

Catches config drift the day it happens, on the Mac, not on a rented GPU.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from querent.models.vit import create_vit
from querent.utils.run_id import run_id

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = str(REPO / "configs")

ARM_NAMES = sorted(p.stem for p in (REPO / "configs" / "arm").glob("*.yaml"))
EXPERIMENTS = sorted(
    str(p.relative_to(REPO / "configs" / "experiment")).removesuffix(".yaml")
    for p in (REPO / "configs" / "experiment").rglob("*.yaml")
)


def _compose(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        return compose(config_name="config", overrides=overrides)


@pytest.mark.parametrize("arm", ARM_NAMES)
def test_arm_composes_and_instantiates(arm):
    cfg = _compose([f"arm={arm}"])
    assert cfg.arm.name == arm
    model_cfg = dict(cfg.model)
    attn_kwargs = dict(cfg.arm.attention.kwargs)
    net = create_vit(model_cfg, cfg.arm.attention.name, attn_kwargs,
                     drop_path_rate=float(cfg.recipe.drop_path_rate))
    assert net.num_tokens == (cfg.model.img_size // cfg.model.patch_size) ** 2


@pytest.mark.parametrize("exp", EXPERIMENTS)
def test_experiment_composes_and_instantiates(exp):
    cfg = _compose([f"experiment={exp}"])
    assert cfg.run_id  # interpolations resolve
    # Nested kwarg overrides (e.g. route: sigmoid merged into the arm's kwargs)
    # must produce a constructible model, not a runtime surprise on a GPU box.
    net = create_vit(dict(cfg.model), cfg.arm.attention.name, dict(cfg.arm.attention.kwargs),
                     drop_path_rate=float(cfg.recipe.drop_path_rate))
    assert net is not None


def test_run_id_yaml_python_consistency():
    cfg = _compose(["arm=a0", "seed=3", "phase=p1"])
    assert str(cfg.run_id) == run_id("p1", "a0", 3)
    cfg2 = _compose(["arm=a0", "seed=3", "phase=p1", "run_suffix=-r2"])
    assert str(cfg2.run_id) == run_id("p1", "a0", 3, "-r2")


def test_run_id_rejects_unsafe():
    with pytest.raises(ValueError):
        run_id("p1", "a 0", 0)
