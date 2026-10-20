# k-dash

`k-dash` implements the frozen v1 design in [`docs/design/`](docs/design/README.md):
content-addressed TVM-FFI CUDA kernel builds, direct OCI Distribution API access,
an atomic local cache, Docker/Nix AOT builds, and synchronous Python loading.

The runnable example is [`examples/axpy`](examples/axpy/README.md). It exports an
AXPY CUDA kernel as one TVM-FFI `kernel.so` and includes an AFS/H20 smoke test.

```bash
python -m pip install -e '.[test]'
pytest
k-dash --help
```

The public Python API is intentionally one function:

```python
import k_dash

module = k_dash.load("examples/axpy", version="1", jit_args={"block_size": 256})
```

Source ignore patterns are relative to the project root. Directory patterns such
as `build`, `build/`, and `build/**` exclude the whole subtree without walking
it. Existing file globs remain supported; required source files cannot be
excluded. Symbolic links are not packaged or traversed.

In an already isolated CI container without a Docker daemon, `k-dash publish
--backend nix ...` builds AOT artifacts directly with installed Nix. This explicit
backend disables Nix's nested sandbox and build-user group; the CI container must
provide isolation. It uses the same frozen release source, locked dependencies,
BuildSpec and binary validation as Docker AOT. Docker remains the default.
Registry credentials are not forwarded to the Nix build subprocess environment.

`publish-build` also accepts `--backend nix` to add an AOT binary to an existing
release without Docker. It builds the source downloaded from that release, not
the local working tree, and leaves the release unchanged. For example, from the
kernel project directory:

```bash
k-dash publish-build --version v0.1.1 --cuda-version 13.0 --cc sm_90a \
  --tvm-ffi-version 0.1.12 --aot-args args.json --backend nix
```

Docker remains the default backend; the same isolated-container requirement
applies to Nix for both publishing commands.
The published release's build recipe must already support the requested target
and TVM-FFI version. If the recipe needs changes, publish a new release instead.

## Releasing to PyPI

Package versions come from `git describe` through `hatch-vcs`. An exact tag such
as `v0.1.0` builds version `0.1.0`; commits after a tag receive a development
version. Build from a Git checkout with its tags available.

Before the first release, configure a [PyPI Trusted Publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
for project `k-dash` (use a pending publisher if the project does not yet exist):

- GitHub owner: `reyoung`
- Repository: `k-dash`
- Workflow: `publish.yml`
- Environment: `pypi`

The workflow uses OIDC, so no PyPI API token secret is required. Push a release
commit and its annotated tag to build, validate, and publish the wheel:

```bash
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin HEAD
git push origin v0.1.0
```

The workflow retains the built wheel as a GitHub Actions artifact and publishes
it only after its metadata and version match the release tag.
