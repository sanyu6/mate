from contextlib import contextmanager
from dataclasses import dataclass
import functools
import importlib
from hashlib import sha256
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlparse

import requests  # type: ignore[import-untyped]

from .arch import Arch, MP31_ARCH, MP32_ARCH


DEFAULT_MUBIN_CACHE_DIR = Path.home() / ".cache" / "mate" / "mubin"
MATE_MUBIN_DOWNLOAD_VERBOSE_ENV = "MATE_MUBIN_DOWNLOAD_VERBOSE"
MATE_MUBIN_MODULES = ("gemm", "flash_attention", "flash_mla", "sage_attention")


@dataclass(frozen=True)
class ArtifactRepository:
    """Repository or local directory that stores MATE MUBIN artifacts."""

    REPOSITORY_ENV: ClassVar[str] = "MATE_MUBIN_REPOSITORY"
    BASE_URL_ENV: ClassVar[str] = "MATE_MUBIN_REPOSITORY_BASE_URL"
    DEFAULT_REPOSITORY: ClassVar[str] = "sw-mate-public-generic"
    DEFAULT_BASE_URL: ClassVar[str] = "https://dl.mthreads.com/repo/repository"

    repository: str = DEFAULT_REPOSITORY
    base_url: str = DEFAULT_BASE_URL

    @classmethod
    def from_env(cls):
        return cls(
            repository=os.environ.get(cls.REPOSITORY_ENV, cls.DEFAULT_REPOSITORY),
            base_url=os.environ.get(cls.BASE_URL_ENV, cls.DEFAULT_BASE_URL),
        )

    @classmethod
    def resolve(cls, repository=None):
        if repository is None:
            return cls.from_env()
        if isinstance(repository, cls):
            return repository
        return cls(
            repository=str(repository),
            base_url=os.environ.get(cls.BASE_URL_ENV, cls.DEFAULT_BASE_URL),
        )

    def url(self, path):
        parsed = urlparse(str(self.repository))
        if parsed.scheme in ("http", "https"):
            base = str(self.repository).rstrip("/")
        else:
            base = f"{self.base_url.rstrip('/')}/{str(self.repository).strip('/')}"
        return f"{base}/{str(path).lstrip('/')}"


@dataclass(frozen=True)
class ArtifactPath:
    """Artifactory paths for MATE MUBIN artifact families."""

    # Existing MP31 releases predate architecture-qualified URLs.  Keep these
    # legacy pins until their payloads are explicitly republished under mp31;
    # all newly published targets use <family>/<arch>/<commit>.
    GEMM: str = "gemm/fbd1a6df72047350033cb4c524dd8145b8f18698/"
    FLASH_ATTENTION: str = "flash_attention/c70384e79ef8ba55f3f7ff12830ca1766d5db4c6/"
    FLASH_MLA: str = "flash_mla/615267cf8e31889221722e80055be37e2b4c2b90/"
    SAGE_ATTENTION: str = "sage_attention/107d06a04203907f26b03449c13754032593b7bb/"

    @classmethod
    @functools.cache
    def for_module(cls, module: str, arch: Arch = MP31_ARCH) -> str:
        return get_artifact_target(module, arch).artifact_path

    @classmethod
    def iter_module_paths(cls):
        for module in MATE_MUBIN_MODULES:
            yield module, cls.for_module(module)


class KernelMapHash:
    """Pinned SHA256 hashes of MATE MUBIN kernel maps."""

    KERNEL_MAP_FILE: str = "kernel_map.json"
    GEMM: str = "ea051c7e23e3ea57eb77140c632097392ceae938ecd6ff3c41272b921e6c8844"
    FLASH_ATTENTION: str = (
        "b70e72e54411328c6cbd753f511d7763b3cf7b5d85a52c9cefdb1bc4bdbb5340"
    )
    FLASH_MLA: str = "875baaee8ec41c09a5ebd4efaed3f1dbc0945be1886e7ea7388d819822ab7a27"
    SAGE_ATTENTION: str = (
        "9369ecee41db0d483a53c5b668df32433cbf8f1a5d6b6c43a330562961685f20"
    )

    @classmethod
    @functools.cache
    def for_module(cls, module: str, arch: Arch = MP31_ARCH) -> str:
        return get_artifact_target(module, arch).kernel_map_sha256


@dataclass(frozen=True)
class ArtifactTarget:
    """Pinned MUBIN artifact selected by module and device architecture."""

    module: str
    arch: Arch
    artifact_path: str
    kernel_map_sha256: str

    @property
    def name(self) -> str:
        return self.module if self.arch == MP31_ARCH else f"{self.module}/{self.arch}"


_ARTIFACT_TARGETS_BY_KEY = {
    (module, MP31_ARCH): ArtifactTarget(
        module,
        MP31_ARCH,
        getattr(ArtifactPath, module.upper()),
        getattr(KernelMapHash, module.upper()),
    )
    for module in MATE_MUBIN_MODULES
}

_ARTIFACT_TARGETS_BY_KEY[("gemm", MP32_ARCH)] = ArtifactTarget(
    "gemm",
    MP32_ARCH,
    "gemm/mp32/3d43e788f83438a99aa61b341ade7412ca0c8ce8/",
    "f1b8ae50224469feeb318e5d2b897a5088a89b09cedaa9c42f459c0993b81c88",
)


def get_artifact_target(module: str, arch: Arch = MP31_ARCH) -> ArtifactTarget:
    try:
        return _ARTIFACT_TARGETS_BY_KEY[(module, arch)]
    except KeyError:
        if module not in MATE_MUBIN_MODULES:
            valid = ", ".join(MATE_MUBIN_MODULES)
            raise ValueError(
                f"Unknown MATE MUBIN module '{module}'. Valid choices: {valid}"
            ) from None
        supported = ", ".join(
            str(target.arch)
            for target in _ARTIFACT_TARGETS_BY_KEY.values()
            if target.module == module
        )
        raise ValueError(
            f"MATE MUBIN module '{module}' does not support architecture "
            f"'{arch}'. Supported architectures: {supported}"
        ) from None


@dataclass(frozen=True)
class KernelMapEntry:
    kernel_name: str
    dispatch_hash: str
    file_hash: str

    @property
    def file_name(self) -> str:
        return f"{self.kernel_name}.o"


@functools.cache
def get_packaged_mubin_dir():
    try:
        mate_mubin = importlib.import_module("mate_mubin")
    except ModuleNotFoundError as exc:
        if exc.name == "mate_mubin":
            return None
        raise

    return Path(mate_mubin.get_mubin_dir())


@functools.cache
def resolve_mubin_module_dir(output_dir, module, artifact_path=None, arch=MP31_ARCH):
    artifact_path = (
        ArtifactPath.for_module(module, arch)
        if artifact_path is None
        else artifact_path
    )
    return Path(output_dir) / str(artifact_path).strip("/")


def resolve_mubin_artifact_root_from_module_dir(
    module, module_dir, artifact_path=None, arch=MP31_ARCH
):
    artifact_path = (
        ArtifactPath.for_module(module, arch)
        if artifact_path is None
        else artifact_path
    )
    root = Path(module_dir)
    for _part in Path(str(artifact_path).strip("/")).parts:
        root = root.parent
    return root


@functools.cache
def apply_sha256(path, _mtime_ns, _size):
    return sha256(Path(path).read_bytes()).hexdigest()


def sha256_file(path):
    path = Path(path).resolve()
    stat = path.stat()
    return apply_sha256(str(path), stat.st_mtime_ns, stat.st_size)


def parse_kernel_map(kernel_map_path):
    with open(kernel_map_path, "r", encoding="utf-8") as f:
        return tuple(KernelMapEntry(**row) for row in json.load(f))


@functools.cache
def _load_kernel_map_cached(kernel_map_path, kernel_map_hash):
    path = Path(kernel_map_path)
    if sha256_file(path) != kernel_map_hash:
        raise RuntimeError(f"Kernel map hash mismatch: {path}")
    return parse_kernel_map(path)


def load_kernel_map(kernel_map_path):
    path = Path(kernel_map_path)
    return _load_kernel_map_cached(str(path.resolve()), sha256_file(path))


@functools.cache
def _env_flag_enabled(name):
    value = os.getenv(name, "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


@contextmanager
def _artifact_file_lock(lock_path):
    from tvm_ffi.utils import FileLock

    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path)):
        yield


def download_file_with_requests(source, destination, timeout=300):
    destination = Path(destination)
    temp_path = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.download"
    )
    verbose = _env_flag_enabled(MATE_MUBIN_DOWNLOAD_VERBOSE_ENV)

    try:
        if verbose:
            print("Downloading MATE MUBIN artifact:")
            print(f"  GET {source}")
        with requests.get(str(source), timeout=timeout) as response:
            response.raise_for_status()
            temp_path.write_bytes(response.content)

        os.replace(temp_path, destination)
        if verbose:
            print("Downloaded MATE MUBIN artifact:")
            print(f"  {source}")
            print(f"  -> {destination}")
        return True
    except (OSError, requests.RequestException) as exc:
        if verbose:
            print("Failed to download MATE MUBIN artifact:")
            print(f"  {source}")
            print(f"  {exc}")
        return False
    finally:
        if temp_path.exists():
            temp_path.unlink()


def download_file(source, destination, retries=4, delay=5, timeout=300):
    """Download an HTTP(S) repository source to destination."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_name(f"{destination.name}.lock")

    with _artifact_file_lock(lock_path):
        for attempt in range(retries):
            if download_file_with_requests(source, destination, timeout=timeout):
                return True
            if attempt < retries - 1:
                time.sleep(delay * (2**attempt))
        return False


@functools.cache
def resolve_mubin_artifact_path(file_name):
    path = Path(file_name)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != file_name:
        raise ValueError(f"Invalid MATE MUBIN artifact path: {file_name}")
    if file_name == KernelMapHash.KERNEL_MAP_FILE:
        return Path(file_name)
    if len(path.parts) == 1 and path.suffix == ".o":
        return Path("mubin") / path.name
    if len(path.parts) == 2 and path.parts[0] == "mubin" and path.suffix == ".o":
        return path
    raise ValueError(f"Unsupported MATE MUBIN artifact: {file_name}")


@functools.cache
def resolve_mubin_output_dir(output_dir=None):
    if output_dir is not None:
        return Path(output_dir)
    env_mubin_dir = os.getenv("MATE_MUBIN_DIR")
    return Path(env_mubin_dir) if env_mubin_dir else DEFAULT_MUBIN_CACHE_DIR


def resolve_active_mubin_dir():
    packaged_mubin_dir = get_packaged_mubin_dir()
    if packaged_mubin_dir is not None:
        return packaged_mubin_dir, "mate-mubin package"

    output_dir = resolve_mubin_output_dir()
    if os.getenv("MATE_MUBIN_DIR"):
        source = "MATE_MUBIN_DIR"
    else:
        source = "default cache"
    return output_dir, source


def download_mubin_module_artifacts(
    module,
    output_dir,
    repository=None,
    artifact_path=None,
    kernel_map_sha256=None,
    arch=MP31_ARCH,
):
    downloaded = download_mubin_module_metadata(
        module,
        output_dir,
        repository=repository,
        artifact_path=artifact_path,
        kernel_map_sha256=kernel_map_sha256,
        arch=arch,
    )
    module_dir = resolve_mubin_module_dir(output_dir, module, artifact_path, arch)
    for entry in load_mubin_module_kernel_map(
        module,
        module_dir,
        kernel_map_sha256=kernel_map_sha256,
        arch=arch,
    ):
        downloaded.append(
            ensure_mubin_kernel_artifact(
                module,
                module_dir,
                entry,
                repository=repository,
                artifact_path=artifact_path,
                kernel_map_sha256=kernel_map_sha256,
                arch=arch,
            )
        )
    return downloaded


def download_mubin_module_metadata(
    module,
    output_dir,
    repository=None,
    artifact_path=None,
    kernel_map_sha256=None,
    arch=MP31_ARCH,
):
    repository = ArtifactRepository.resolve(repository)
    target = get_artifact_target(module, arch)
    artifact_path = target.artifact_path if artifact_path is None else artifact_path
    expected_hash = (
        target.kernel_map_sha256 if kernel_map_sha256 is None else kernel_map_sha256
    )
    module_dir = resolve_mubin_module_dir(output_dir, module, artifact_path, arch)
    destination = module_dir / KernelMapHash.KERNEL_MAP_FILE
    source = repository.url(
        f"{str(artifact_path).strip('/')}/{KernelMapHash.KERNEL_MAP_FILE}"
    )
    if not download_file(source, destination):
        raise RuntimeError(
            f"Failed to download kernel map for MATE MUBIN {target.name}"
        )
    verify_mubin_module_metadata.cache_clear()
    if (
        arch == MP32_ARCH or not os.getenv("MATE_MUBIN_VERIFY_DISABLED")
    ) and sha256_file(destination) != expected_hash:
        raise RuntimeError(f"Kernel map hash mismatch for MATE MUBIN {target.name}")
    return [destination]


def download_mubin_module_runtime_artifacts(
    module,
    output_dir,
    repository=None,
    artifact_path=None,
    kernel_map_sha256=None,
    arch=MP31_ARCH,
):
    """Compatibility alias for metadata-only module bootstrap downloads."""
    return download_mubin_module_metadata(
        module,
        output_dir,
        repository=repository,
        artifact_path=artifact_path,
        kernel_map_sha256=kernel_map_sha256,
        arch=arch,
    )


@functools.cache
def verify_mubin_module_metadata(
    module, module_dir, kernel_map_sha256=None, arch=MP31_ARCH
):
    module_dir = Path(module_dir)
    expected_hash = (
        KernelMapHash.for_module(module, arch)
        if kernel_map_sha256 is None
        else kernel_map_sha256
    )
    kernel_map_path = module_dir / KernelMapHash.KERNEL_MAP_FILE
    if not kernel_map_path.is_file():
        return False
    return not (
        (arch == MP32_ARCH or not os.getenv("MATE_MUBIN_VERIFY_DISABLED"))
        and sha256_file(kernel_map_path) != expected_hash
    )


def load_mubin_module_kernel_map(
    module, module_dir, kernel_map_sha256=None, arch=MP31_ARCH
):
    if not verify_mubin_module_metadata(module, module_dir, kernel_map_sha256, arch):
        raise RuntimeError(
            f"MATE MUBIN kernel map for {module}/{arch} failed verification"
        )
    return load_kernel_map(Path(module_dir) / KernelMapHash.KERNEL_MAP_FILE)


def verify_mubin_module_artifacts(
    module, module_dir, kernel_map_sha256=None, arch=MP31_ARCH
):
    try:
        entries = load_mubin_module_kernel_map(
            module,
            module_dir,
            kernel_map_sha256=kernel_map_sha256,
            arch=arch,
        )
    except RuntimeError:
        return False
    verify_disabled = arch != MP32_ARCH and os.getenv("MATE_MUBIN_VERIFY_DISABLED")
    for entry in entries:
        path = Path(module_dir) / "mubin" / entry.file_name
        if not path.is_file():
            return False
        if not verify_disabled and sha256_file(path) != entry.file_hash:
            return False
    return True


def verify_mubin_module_runtime_artifacts(
    module, module_dir, kernel_map_sha256=None, arch=MP31_ARCH
):
    """Compatibility alias for metadata-only module verification."""
    return verify_mubin_module_metadata(
        module,
        module_dir,
        kernel_map_sha256=kernel_map_sha256,
        arch=arch,
    )


def _require_packaged_mubin_metadata(
    target: ArtifactTarget,
    module_dir: Path,
    kernel_map_sha256: str,
) -> None:
    kernel_map_path = module_dir / KernelMapHash.KERNEL_MAP_FILE
    if not kernel_map_path.is_file():
        raise RuntimeError(
            f"Installed mate-mubin package is missing {target.name} metadata: "
            f"{kernel_map_path}"
        )
    if sha256_file(kernel_map_path) != kernel_map_sha256:
        raise RuntimeError(
            f"Installed mate-mubin package contains corrupt {target.name} "
            f"metadata: {kernel_map_path}"
        )


def ensure_mubin_kernel_artifact(
    module,
    module_dir,
    entry,
    repository=None,
    artifact_path=None,
    kernel_map_sha256=None,
    arch=MP31_ARCH,
):
    target = get_artifact_target(module, arch)
    artifact_path = target.artifact_path if artifact_path is None else artifact_path
    expected_map_hash = (
        target.kernel_map_sha256 if kernel_map_sha256 is None else kernel_map_sha256
    )
    module_dir = Path(module_dir)
    artifact_rel = resolve_mubin_artifact_path(entry.file_name)
    destination = module_dir / artifact_rel
    packaged_mubin_dir = get_packaged_mubin_dir()
    is_packaged_module = (
        packaged_mubin_dir is not None
        and module_dir.resolve().is_relative_to(packaged_mubin_dir.resolve())
    )
    if is_packaged_module and arch == MP31_ARCH and destination.is_file():
        return destination
    if is_packaged_module and arch == MP32_ARCH:
        _require_packaged_mubin_metadata(target, module_dir, expected_map_hash)
        if not destination.is_file():
            raise RuntimeError(
                f"Installed mate-mubin package is missing {target.name} artifact: "
                f"{entry.file_name}"
            )
        if sha256_file(destination) != entry.file_hash:
            raise RuntimeError(
                f"Installed mate-mubin package contains corrupt {target.name} "
                f"artifact: {entry.file_name}"
            )
        return destination

    mubin_root = resolve_mubin_artifact_root_from_module_dir(
        module, module_dir, artifact_path, arch
    )
    if not verify_mubin_module_metadata(module, module_dir, expected_map_hash, arch):
        if os.getenv("MATE_MUBIN_NO_DOWNLOAD"):
            raise RuntimeError(
                f"MATE MUBIN metadata for {target.name} is not available locally"
            )
        download_mubin_module_metadata(
            module,
            mubin_root,
            repository=repository,
            artifact_path=artifact_path,
            kernel_map_sha256=expected_map_hash,
            arch=arch,
        )

    verify_disabled = arch != MP32_ARCH and os.getenv("MATE_MUBIN_VERIFY_DISABLED")
    verbose = _env_flag_enabled(MATE_MUBIN_DOWNLOAD_VERBOSE_ENV)
    if destination.is_file():
        if verify_disabled or sha256_file(destination) == entry.file_hash:
            if verbose:
                print(f"Using local MATE MUBIN kernel for {target.name}: {destination}")
            return destination
        if verbose:
            print(f"Refreshing MATE MUBIN kernel for {target.name}: {destination}")

    if os.getenv("MATE_MUBIN_NO_DOWNLOAD"):
        raise RuntimeError(
            f"MATE MUBIN artifact is not available locally: {entry.file_name}"
        )

    if verbose:
        print(f"Downloading MATE MUBIN kernel for {target.name}: {entry.file_name}")
    repository = ArtifactRepository.resolve(repository)
    source_path = f"{str(artifact_path).strip('/')}/{artifact_rel.as_posix()}"
    if not download_file(repository.url(source_path), destination):
        raise RuntimeError(f"Failed to download MATE MUBIN artifact: {entry.file_name}")
    if not verify_disabled and sha256_file(destination) != entry.file_hash:
        raise RuntimeError(f"Hash mismatch for MATE MUBIN artifact: {entry.file_name}")
    return destination


@functools.cache
def ensure_mubin_module_artifacts(
    module, cache_dir=None, repository=None, arch=MP31_ARCH
):
    """Ensure module metadata; kernels are downloaded lazily."""

    verbose = _env_flag_enabled(MATE_MUBIN_DOWNLOAD_VERBOSE_ENV)
    target = get_artifact_target(module, arch)
    packaged_mubin_dir = get_packaged_mubin_dir()
    if packaged_mubin_dir is not None:
        packaged_mubin_dir = packaged_mubin_dir.resolve()
        module_dir = resolve_mubin_module_dir(
            packaged_mubin_dir, module, target.artifact_path, arch
        )
        if arch == MP32_ARCH:
            _require_packaged_mubin_metadata(
                target, module_dir, target.kernel_map_sha256
            )
        if verbose:
            print(
                f"Using installed mate-mubin artifacts for {target.name}: {module_dir}"
            )
        return module_dir

    cache_root = resolve_mubin_output_dir(cache_dir)
    module_dir = resolve_mubin_module_dir(
        cache_root, module, target.artifact_path, arch
    )
    if verify_mubin_module_metadata(module, module_dir, target.kernel_map_sha256, arch):
        if verbose:
            print(f"Using local MATE MUBIN metadata for {target.name}: {module_dir}")
        return module_dir

    if os.getenv("MATE_MUBIN_NO_DOWNLOAD"):
        if verbose:
            print(f"MATE MUBIN artifacts for {target.name} are not available locally")
        raise RuntimeError(
            f"MATE MUBIN artifacts for {target.name} are not available locally"
        )

    if verbose:
        print(f"Downloading MATE MUBIN metadata for {target.name} into {cache_root}")
    download_mubin_module_metadata(
        module,
        cache_root,
        repository=repository,
        artifact_path=target.artifact_path,
        kernel_map_sha256=target.kernel_map_sha256,
        arch=arch,
    )
    if verbose:
        print(f"Using downloaded MATE MUBIN metadata for {target.name}: {module_dir}")
    return module_dir


def download_mubin_artifacts(output_dir, modules=MATE_MUBIN_MODULES, repository=None):
    output_dir = resolve_mubin_output_dir(output_dir)
    downloaded = []
    for module in modules:
        downloaded.extend(
            download_mubin_module_artifacts(
                module,
                output_dir,
                repository=repository,
            )
        )
    return downloaded


def download_artifacts(output_dir=None, modules=MATE_MUBIN_MODULES, repository=None):
    return download_mubin_artifacts(output_dir, modules=modules, repository=repository)


def get_mubin_artifacts_status(output_dir=None, modules=MATE_MUBIN_MODULES):
    output_dir = resolve_mubin_output_dir(output_dir)
    statuses = []
    for module in modules:
        module_dir = resolve_mubin_module_dir(output_dir, module)
        if verify_mubin_module_artifacts(module, module_dir):
            status = "Downloaded"
        elif verify_mubin_module_metadata(module, module_dir):
            status = "Metadata only"
        elif module_dir.exists():
            status = "Incomplete"
        else:
            status = "Missing"
        statuses.append((module, status, module_dir))
    return tuple(statuses)


def get_active_mubin_artifacts_status(modules=MATE_MUBIN_MODULES):
    active_dir, source = resolve_active_mubin_dir()
    return active_dir, source, get_mubin_artifacts_status(active_dir, modules)


def clear_mubin_artifacts(output_dir=None):
    output_dir = resolve_mubin_output_dir(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
        verify_mubin_module_metadata.cache_clear()
        ensure_mubin_module_artifacts.cache_clear()
        return output_dir, True
    return output_dir, False
