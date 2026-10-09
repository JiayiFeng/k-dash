from types import SimpleNamespace
import sys

import pytest

from k_dash.builder import docker_aot
from k_dash.model import BuildSpec, TargetSpec
from k_dash.templates import init_project


def test_docker_builder_bounds_nofile_before_start(tmp_path, monkeypatch):
    root = tmp_path / 'kernel'
    init_project('owner/kernel', root, 'cpp', None)
    class ContainerCreated(Exception): pass
    def create(*args, **kwargs):
        limits = kwargs['ulimits']
        assert len(limits) == 1
        assert (limits[0].name, limits[0].soft, limits[0].hard) == ('nofile', 65536, 65536)
        assert kwargs['volumes'] == {'k-dash-nix-store': {'bind': '/nix', 'mode': 'rw'}}
        assert '--max-jobs' not in kwargs['command'][2]
        raise ContainerCreated
    client = SimpleNamespace(volumes=SimpleNamespace(get=lambda _: None),
                             containers=SimpleNamespace(create=create))
    monkeypatch.setitem(sys.modules, 'docker', SimpleNamespace(
        from_env=lambda: client, types=SimpleNamespace(Ulimit=lambda **kwargs: SimpleNamespace(**kwargs))))
    spec = BuildSpec('sha256:' + '0' * 64, {}, TargetSpec('linux', 'x86_64', '13.0', 'sm_90a').as_dict())
    with pytest.raises(ContainerCreated):
        docker_aot(root, spec)


def test_nix_aot_does_not_forward_registry_credentials(tmp_path, monkeypatch):
    from k_dash.builder import nix_aot
    import subprocess
    monkeypatch.setenv("REGISTRY_PASSWORD", "secret")
    monkeypatch.setenv("DOCKER_CONFIG", "/private/docker")
    monkeypatch.setenv("K_DASH_NIX_MAX_JOBS", "2")
    monkeypatch.setattr("k_dash.builder.shutil.which", lambda _: "/usr/bin/nix")
    output = tmp_path / "output"
    output.mkdir()
    (output / "kernel.so").write_bytes(b"ELF")
    def run(command, **kwargs):
        assert "REGISTRY_PASSWORD" not in kwargs["env"]
        assert "DOCKER_CONFIG" not in kwargs["env"]
        assert "sandbox = false" in kwargs["env"]["NIX_CONFIG"]
        assert command[command.index("--max-jobs") + 1] == "2"
        return subprocess.CompletedProcess(command, 0, str(output) + "\n")
    monkeypatch.setattr("k_dash.builder.subprocess.run", run)
    spec = BuildSpec('sha256:' + '0' * 64, {}, TargetSpec('linux', 'x86_64', '13.0', 'sm_90a').as_dict())
    data, provenance = nix_aot(tmp_path, spec)
    assert data == b"ELF"
    assert provenance["mode"] == "nix-aot"


def _spec() -> BuildSpec:
    return BuildSpec('sha256:' + '0' * 64, {}, TargetSpec('linux', 'x86_64', '13.0', 'sm_90a').as_dict())


def test_build_jobs_forward_to_nix_max_jobs(tmp_path, monkeypatch):
    import subprocess
    from k_dash.builder import docker_aot, local_jit, nix_aot
    from k_dash.cli import main
    from k_dash.errors import KDashError

    monkeypatch.setenv("K_DASH_NIX_MAX_JOBS", "1")
    monkeypatch.setattr("k_dash.builder.shutil.which", lambda _: "/usr/bin/nix")
    output = tmp_path / "output"
    output.mkdir()
    (output / "kernel.so").write_bytes(b"ELF")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, str(output) + "\n")

    monkeypatch.setattr("k_dash.builder.subprocess.run", run)
    local_jit(tmp_path, _spec(), max_jobs="8")
    nix_aot(tmp_path, _spec(), max_jobs="auto")
    assert [command[command.index("--max-jobs") + 1] for command in commands] == ["8", "auto"]

    class ContainerCreated(Exception):
        pass

    def create(*args, **kwargs):
        assert "--max-jobs 8 --file" in kwargs["command"][2]
        raise ContainerCreated

    root = tmp_path / "kernel"
    init_project("owner/kernel", root, "cpp", None)
    client = SimpleNamespace(
        volumes=SimpleNamespace(get=lambda _: None),
        containers=SimpleNamespace(create=create),
    )
    monkeypatch.setitem(sys.modules, "docker", SimpleNamespace(
        from_env=lambda: client,
        types=SimpleNamespace(Ulimit=lambda **kwargs: SimpleNamespace(**kwargs)),
    ))
    with pytest.raises(ContainerCreated):
        docker_aot(root, _spec(), max_jobs="8")
    with pytest.raises(KDashError, match="non-negative integer or auto"):
        docker_aot(root, _spec(), max_jobs="8;touch")

    seen = {}

    def build(*_args, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr("k_dash.cli.load_project", lambda *_: None)
    monkeypatch.setattr("k_dash.cli.local_build", build)
    assert main(["build", "--version", "dev", "--cuda-version", "12.8", "--cc", "sm_90a", "-j", "4"]) == 0
    assert seen["max_jobs"] == "4"
    assert main(["build", "--version", "dev", "--cuda-version", "12.8", "--cc", "sm_90a", "--jobs", "auto"]) == 0
    assert seen["max_jobs"] == "auto"
    with pytest.raises(SystemExit):
        main(["build", "--version", "dev", "--cuda-version", "12.8", "--cc", "sm_90a", "-j", "nope"])
