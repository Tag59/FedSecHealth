import torch
from flwr.app import Context, RecordDict
from flwr.serverapp.strategy import FedAvg, FedMedian, FedTrimmedAvg, Krum, MultiKrum

from fedsechealth.flower import client_app
from fedsechealth.flower.common import dataset, fl_config, model_fn
from fedsechealth.flower.server_app import build_strategy

RUN = {
    "num-hospitals": 3,
    "num-server-rounds": 2,
    "dataset": "breast_cancer",
    "model": "mlp",
    "local-epochs": 1,
    "batch-size": 32,
    "learning-rate": 0.1,
    "seed": 0,
}


def _context(run: dict, state: RecordDict | None = None) -> Context:
    node = {"partition-id": 1, "num-partitions": 3}
    return Context(
        run_id=1, node_id=7, node_config=node, state=state or RecordDict(), run_config=run
    )


def _global_state(run: dict) -> dict:
    torch.manual_seed(0)
    return model_fn(run, dataset(run, 3))().state_dict()


def test_run_config_is_parsed_into_fl_config():
    cfg = fl_config({**RUN, "dp-enabled": True, "dp-epsilon": 3.0}, seed=5)
    assert cfg.rounds == 2 and cfg.seed == 5 and cfg.model == "mlp"
    assert cfg.dp.enabled and cfg.dp.target_epsilon == 3.0


def test_strategies_map_to_flower_builtins():
    assert type(build_strategy("fedavg", 5, 0, 0.2)) is FedAvg
    assert isinstance(build_strategy("median", 5, 0, 0.2), FedMedian)
    assert isinstance(build_strategy("trimmed_mean", 5, 0, 0.2), FedTrimmedAvg)
    assert isinstance(build_strategy("krum", 5, 1, 0.2), Krum)
    assert isinstance(build_strategy("multi_krum", 5, 1, 0.2), MultiKrum)


def test_client_trains_only_on_its_partition_and_returns_weights():
    state, metrics = client_app.local_train(_global_state(RUN), _context(RUN))
    assert metrics["num-examples"] == len(dataset(RUN, 3).clients[1])
    assert 0.5 < metrics["train-accuracy"] <= 1.0
    assert set(state) == set(_global_state(RUN))


def test_dp_budget_accumulates_across_messages():
    run = {**RUN, "dp-enabled": True, "dp-epsilon": 8.0}
    ctx = _context(run)
    eps = []
    for _ in range(2):  # a fresh Hospital per message, as with a new ClientApp process
        _, metrics = client_app.local_train(_global_state(run), ctx)
        eps.append(metrics["epsilon"])
    assert 0 < eps[0] < eps[1] <= 8.0 + 1e-3
