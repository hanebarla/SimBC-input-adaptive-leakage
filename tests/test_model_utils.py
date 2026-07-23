import json

from robo_manip_baselines.sarnn.lib.model_utils import load_experiment_args


def test_legacy_args_receive_public_experiment_defaults(tmp_path):
    args_path = tmp_path / "args.json"
    args_path.write_text(
        json.dumps({"qcfs": True, "rec_dim": 50, "k_dim": 20}),
        encoding="utf-8",
    )
    params = load_experiment_args(args_path)
    assert params["im_size"] == [128, 128]
    assert params["qcfs_L"] == 8
    assert params["compile"] is False
    assert params["wobn"] is True
