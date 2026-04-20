from analysis.classical_models import _resolve_model_names
from analysis.Net import build_torch_model, get_model_spec, list_models


def test_resolve_feature_classifier_can_select_one_model() -> None:
    assert _resolve_model_names("rbf_svm", ("logistic_regression", "linear_svm")) == ["rbf_svm"]


def test_resolve_feature_classifier_all_uses_configured_known_models() -> None:
    assert _resolve_model_names("all", ("linear_svm", "unknown_model", "random_forest")) == [
        "linear_svm",
        "random_forest",
    ]


def test_resolve_deep_network_can_select_one_model() -> None:
    assert get_model_spec("BiDANN").protocol == "transductive_da"


def test_libeer_registry_contains_expected_models() -> None:
    names = {spec.name for spec in list_models()}
    assert {"EEGNet", "DGCNN", "STRNN", "BiDANN", "MsMDA", "PRRL"}.issubset(names)


def test_all_registered_torch_models_build_for_32_channel_input() -> None:
    for spec in list_models():
        build_torch_model(
            spec.name,
            n_channels=32,
            n_classes=3,
            n_samples=128,
            n_bands=5,
            sequence_length=9,
            num_domains=2,
        )


def test_build_torch_models_for_key_input_families() -> None:
    import torch

    raw_model = build_torch_model("EEGNet", n_channels=32, n_classes=3, n_samples=128)
    de_model = build_torch_model("DGCNN", n_channels=32, n_classes=3, n_bands=5)
    seq_model = build_torch_model("STRNN", n_channels=32, n_classes=3, n_bands=5, sequence_length=9)
    da_model = build_torch_model("BiDANN", n_channels=32, n_classes=3, n_bands=5, sequence_length=9)

    assert raw_model(torch.randn(2, 32, 128)).shape == (2, 3)
    assert de_model(torch.randn(2, 32, 5)).shape == (2, 3)
    assert seq_model(torch.randn(2, 9, 32, 5)).shape == (2, 3)
    assert da_model(torch.randn(2, 9, 32, 5), torch.randn(2, 9, 32, 5))["logits"].shape == (2, 3)


def test_raw_model_can_build_binary_single_logit_head() -> None:
    import torch

    model = build_torch_model("FBSTCNet", n_channels=32, n_classes=1, n_samples=800)
    assert model(torch.randn(2, 32, 800)).shape == (2, 1)
