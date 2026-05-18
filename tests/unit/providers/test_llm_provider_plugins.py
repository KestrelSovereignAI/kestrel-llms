from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[3]
for src in [
    ROOT / "providers/kestrel-llm-deepseek/src",
    ROOT / "providers/kestrel-llm-xai/src",
    ROOT / "providers/kestrel-llm-kimi/src",
    ROOT / "providers/kestrel-llm-llama-cpp/src",
    ROOT / "providers/kestrel-llm-openai-compat/src",
]:
    sys.path.insert(0, str(src))

from kestrel_sovereign.llm.provider_registry import (  # noqa: E402
    LLM_PROVIDER_ENTRY_POINT_GROUP,
    ProviderRegistry,
)


def _entry_point(name: str, cls):
    ep = MagicMock()
    ep.name = name
    ep.value = f"{cls.__module__}:{cls.__name__}"
    ep.load.return_value = cls
    return ep


def _patch_entry_points(entry_points: list):
    eps = MagicMock()
    eps.select.return_value = entry_points
    return patch("kestrel_sovereign.entrypoints.importlib.metadata.entry_points", return_value=eps)


@pytest.mark.parametrize(
    ("module_name", "class_name", "vendor", "env_var", "base_url", "default_model"),
    [
        ("kestrel_llm_deepseek", "DeepSeekAdapter", "deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-v4-flash"),
        ("kestrel_llm_xai", "XAIAdapter", "xai", "XAI_API_KEY", "https://api.x.ai/v1", "grok-4.3"),
        ("kestrel_llm_kimi", "KimiAdapter", "kimi", "MOONSHOT_API_KEY", "https://api.moonshot.ai/v1", "kimi-k2.6"),
    ],
)
def test_cloud_provider_factories_build_route_info(
    monkeypatch,
    module_name,
    class_name,
    vendor,
    env_var,
    base_url,
    default_model,
):
    module = __import__(module_name)
    cls = getattr(module, class_name)
    monkeypatch.setenv(env_var, "sk-test")
    fake_client = MagicMock()

    with patch.object(module.openai, "AsyncOpenAI", return_value=fake_client) as async_openai:
        info = cls.create_provider({})

    async_openai.assert_called_once_with(api_key="sk-test", base_url=base_url, max_retries=0)
    assert info.name == f"{vendor}:api"
    assert info.vendor == vendor
    assert info.route == "api"
    assert info.client is fake_client
    assert info.model == default_model
    assert info.is_cloud is True
    assert info.is_local is False


def test_kimi_accepts_kimi_api_key_alias(monkeypatch):
    import kestrel_llm_kimi

    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.setenv("KIMI_API_KEY", "sk-kimi")

    with patch.object(kestrel_llm_kimi.openai, "AsyncOpenAI") as async_openai:
        kestrel_llm_kimi.KimiAdapter.create_provider({})

    assert async_openai.call_args.kwargs["api_key"] == "sk-kimi"


def test_llama_cpp_factory_is_local_without_api_key(monkeypatch):
    import kestrel_llm_llama_cpp

    monkeypatch.delenv("LLAMA_CPP_API_KEY", raising=False)
    monkeypatch.delenv("LLAMA_CPP_BASE_URL", raising=False)
    fake_client = MagicMock()

    with patch.object(kestrel_llm_llama_cpp.openai, "AsyncOpenAI", return_value=fake_client) as async_openai:
        info = kestrel_llm_llama_cpp.LlamaCppAdapter.create_provider({})

    async_openai.assert_called_once_with(
        api_key="local",
        base_url="http://localhost:8000/v1",
        max_retries=0,
    )
    assert info.name == "llama_cpp:local"
    assert info.vendor == "llama_cpp"
    assert info.route == "local"
    assert info.is_cloud is False
    assert info.is_local is True


def test_registry_discovers_first_wave_plugins(monkeypatch):
    import kestrel_llm_deepseek
    import kestrel_llm_kimi
    import kestrel_llm_llama_cpp
    import kestrel_llm_xai

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.setenv("XAI_API_KEY", "sk-xai")
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-kimi")

    entry_points = [
        _entry_point("deepseek", kestrel_llm_deepseek.DeepSeekAdapter),
        _entry_point("xai", kestrel_llm_xai.XAIAdapter),
        _entry_point("kimi", kestrel_llm_kimi.KimiAdapter),
        _entry_point("llama_cpp", kestrel_llm_llama_cpp.LlamaCppAdapter),
    ]
    config = {
        "route_priority": ["deepseek:api", "xai:api", "kimi:api", "llama_cpp:local"],
        "vendors": {
            "deepseek": {"routes": {"api": {}}},
            "xai": {"routes": {"api": {}}},
            "kimi": {"routes": {"api": {}}},
            "llama_cpp": {"is_cloud": False, "routes": {"local": {}}},
        },
    }

    with (
        _patch_entry_points(entry_points),
        patch.object(kestrel_llm_deepseek.openai, "AsyncOpenAI", return_value=MagicMock()),
        patch.object(kestrel_llm_xai.openai, "AsyncOpenAI", return_value=MagicMock()),
        patch.object(kestrel_llm_kimi.openai, "AsyncOpenAI", return_value=MagicMock()),
        patch.object(kestrel_llm_llama_cpp.openai, "AsyncOpenAI", return_value=MagicMock()),
    ):
        providers = ProviderRegistry(config).initialize_providers()

    assert [provider.name for provider in providers] == [
        "deepseek:api",
        "xai:api",
        "kimi:api",
        "llama_cpp:local",
    ]
    assert {provider.name for provider in providers if provider.is_local} == {"llama_cpp:local"}
    assert {provider.name for provider in providers if provider.is_cloud} == {
        "deepseek:api",
        "xai:api",
        "kimi:api",
    }


def test_entry_point_group_stays_stable():
    assert LLM_PROVIDER_ENTRY_POINT_GROUP == "kestrel_sovereign.llm_providers"


def _load_pyproject(package: str) -> dict:
    with (ROOT / "providers" / package / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


@pytest.mark.parametrize(
    ("package", "entry_point"),
    [
        ("kestrel-llm-deepseek", "deepseek"),
        ("kestrel-llm-xai", "xai"),
        ("kestrel-llm-kimi", "kimi"),
        ("kestrel-llm-llama-cpp", "llama_cpp"),
    ],
)
def test_provider_pyprojects_keep_plugin_contract(package, entry_point):
    pyproject = _load_pyproject(package)
    project = pyproject["project"]

    assert project["name"] == package
    assert "kestrel-sovereign-sdk>=0.14,<1" in project["dependencies"]
    assert "kestrel-llm-openai-compat==0.1.0" in project["dependencies"]
    assert not any(dep.startswith("kestrel_sovereign") for dep in project["dependencies"])

    entry_points = pyproject["project"]["entry-points"][LLM_PROVIDER_ENTRY_POINT_GROUP]
    assert list(entry_points) == [entry_point]


def test_meta_package_extras_track_first_wave_packages():
    pyproject = _load_pyproject("kestrel-llms")
    extras = pyproject["project"]["optional-dependencies"]

    assert set(extras) == {"deepseek", "xai", "kimi", "llama-cpp", "cloud", "local", "all"}
    assert extras["deepseek"] == ["kestrel-llm-deepseek>=0.1.0,<0.2"]
    assert extras["xai"] == ["kestrel-llm-xai>=0.1.0,<0.2"]
    assert extras["kimi"] == ["kestrel-llm-kimi>=0.1.0,<0.2"]
    assert extras["llama-cpp"] == ["kestrel-llm-llama-cpp>=0.1.0,<0.2"]
    assert set(extras["cloud"]) == {
        "kestrel-llm-deepseek>=0.1.0,<0.2",
        "kestrel-llm-xai>=0.1.0,<0.2",
        "kestrel-llm-kimi>=0.1.0,<0.2",
    }
    assert extras["local"] == ["kestrel-llm-llama-cpp>=0.1.0,<0.2"]
    assert set(extras["all"]) == set(extras["cloud"]) | set(extras["local"])


def test_shared_openai_compat_package_has_no_entry_point():
    pyproject = _load_pyproject("kestrel-llm-openai-compat")
    project = pyproject["project"]

    assert project["name"] == "kestrel-llm-openai-compat"
    assert "kestrel-sovereign-sdk>=0.14,<1" in project["dependencies"]
    assert "entry-points" not in pyproject["project"]
