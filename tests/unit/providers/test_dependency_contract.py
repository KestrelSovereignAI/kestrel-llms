"""Published provider contracts resolve without Core or private SDK sources."""

from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version
import pytest


ROOT = Path(__file__).resolve().parents[3]
GROUP = "kestrel_sovereign.llm_providers"
PROVIDERS = {"deepseek": "deepseek", "xai": "xai", "kimi": "kimi", "vertex": "vertex_ai", "bedrock": "bedrock"}
PACKAGES = ["kestrel-llm-" + name for name in [*PROVIDERS, "openai-compat"]] + ["kestrel-llms"]


def manifest(package=None):
    path = ROOT / "pyproject.toml" if package is None else ROOT / "providers" / package / "pyproject.toml"
    return tomllib.loads(path.read_text())


def requirements(project):
    return {canonicalize_name(r.name): r for r in map(Requirement, project["dependencies"])}


def requires_floor(requirement, minimum):
    assert any(s.operator == ">=" and Version(s.version) >= Version(minimum) for s in requirement.specifier)


@pytest.mark.parametrize("package", PACKAGES)
def test_every_published_package_requires_the_completion_contract(package):
    project = manifest(package)["project"]
    assert project["name"] == package
    assert project["version"] == "0.2.0"
    requires_floor(requirements(project)["kestrel-sovereign-sdk"], "0.49.0")


@pytest.mark.parametrize("package", [None, *PACKAGES])
def test_core_is_not_a_production_or_workspace_dependency(package):
    assert "kestrel-sovereign" not in requirements(manifest(package)["project"])


@pytest.mark.parametrize("provider,entry_point", PROVIDERS.items())
def test_provider_metadata_preserves_its_entry_point_and_tested_native_floor(provider, entry_point):
    project = manifest("kestrel-llm-" + provider)["project"]
    deps = requirements(project)
    assert list(project["entry-points"][GROUP]) == [entry_point]
    if provider in {"deepseek", "xai", "kimi"}:
        requires_floor(deps["openai"], "3.46.0")
        requires_floor(deps["kestrel-llm-openai-compat"], "0.2.0")
    elif provider == "vertex":
        requires_floor(deps["google-genai"], "2.34.0")
        requires_floor(deps["httpx2"], "2.16.0")
        requires_floor(deps["google-auth"], "2.62.1")
    else:
        requires_floor(deps["boto3"], "1.43.124")
        requires_floor(deps["botocore"], "1.43.124")
        requires_floor(deps["urllib3"], "2.8.0")


def test_shared_helper_has_no_provider_entry_point():
    assert "entry-points" not in manifest("kestrel-llm-openai-compat")["project"]


def test_meta_extras_require_current_providers_without_the_old_ceiling():
    extras = manifest("kestrel-llms")["project"]["optional-dependencies"]
    assert set(extras) == {*PROVIDERS, "cloud", "all"}
    expected = {"kestrel-llm-" + name for name in PROVIDERS}
    for alias, entries in extras.items():
        deps = {canonicalize_name(r.name): r for r in map(Requirement, entries)}
        assert set(deps) == (expected if alias in {"cloud", "all"} else {"kestrel-llm-" + alias})
        for requirement in deps.values():
            requires_floor(requirement, "0.2.0")
            assert "0.1.99" not in requirement.specifier
            assert "0.3.0" in requirement.specifier


def test_workspace_installs_every_vendor_sdk_through_its_code_packages():
    data = manifest()
    expected = set(PACKAGES) - {"kestrel-llms"}
    assert set(requirements(data["project"])) == expected
    assert set(data["tool"]["uv"]["sources"]) == expected
    assert all(source == {"workspace": True} for source in data["tool"]["uv"]["sources"].values())
    assert "override-dependencies" not in data["tool"]["uv"]


def test_lock_uses_the_normal_public_sdk_without_core():
    packages = tomllib.loads((ROOT / "uv.lock").read_text())["package"]
    assert not any(canonicalize_name(p["name"]) == "kestrel-sovereign" for p in packages)
    sdk, = [p for p in packages if canonicalize_name(p["name"]) == "kestrel-sovereign-sdk"]
    assert Version(sdk["version"]) >= Version("0.49.0")
    assert sdk["source"] == {"registry": "https://pypi.org/simple"}
    urllib3, = [p for p in packages if canonicalize_name(p["name"]) == "urllib3"]
    assert Version(urllib3["version"]) >= Version("2.8.0")
