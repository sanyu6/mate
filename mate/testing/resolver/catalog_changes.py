"""Conservatively compare static catalog registrations across Git snapshots."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .catalog import OperationCatalog


@dataclass(frozen=True, slots=True)
class _Registration:
    interfaces: tuple[str, ...]
    tests: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Source:
    registrations: dict[str, _Registration]
    structure: tuple[str, ...]


def changed_catalog_tests(
    repository: Path,
    catalog: OperationCatalog,
    snapshots: Sequence[Path],
) -> dict[str, tuple[str, ...]] | None:
    """Return affected tests, or None when a catalog edit cannot be isolated."""
    if len(catalog.definition_files) != 1:
        return None
    definition = catalog.definition_files[0]
    sources: list[_Source] = []
    for snapshot in snapshots:
        try:
            source = _parse_source((snapshot / definition).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, SyntaxError, ValueError):
            return None
        if source is None:
            return None
        sources.append(source)

    current = sources[-1].registrations
    if tuple(sorted(current)) != catalog.names or any(
        current[name].tests
        != tuple(path.as_posix() for path in catalog.get(name).tests)
        for name in catalog.names
    ):
        return None
    if any(source.structure != sources[0].structure for source in sources[1:]):
        return None

    changed: dict[str, set[str]] = {}
    for before, after in zip(sources, sources[1:]):
        if before.registrations.keys() - after.registrations.keys():
            return None
        for name in before.registrations.keys() | after.registrations.keys():
            old = before.registrations.get(name)
            new = after.registrations.get(name)
            if old == new:
                continue
            tests = changed.setdefault(name, set())
            tests.update(new.tests if new else ())
            if old:
                tests.update(
                    test for test in old.tests if (repository / test).is_file()
                )
    return {name: tuple(sorted(tests)) for name, tests in sorted(changed.items())}


def _parse_source(source: str) -> _Source | None:
    tree = ast.parse(source)
    registrations: dict[str, _Registration] = {}
    structure: list[str] = []
    for statement in tree.body:
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and _is_catalog_registration(statement)
        ):
            registration = _parse_registration(statement.value)
            if registration is None or registration[0] in registrations:
                return None
            name, spec = registration
            registrations[name] = spec
            if not structure or structure[-1] != "catalog registrations":
                structure.append("catalog registrations")
        elif isinstance(statement, ast.Assign) and _is_policy_assignment(statement):
            if not _valid_policy(statement.value):
                return None
            structure.append("selection policy")
        else:
            structure.append(ast.dump(statement, include_attributes=False))
    return _Source(registrations, tuple(structure))


def _is_catalog_registration(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and isinstance(statement.value.func.value, ast.Name)
        and statement.value.func.value.id == "TEST_CATALOG"
        and statement.value.func.attr == "register"
    )


def _parse_registration(call: ast.Call) -> tuple[str, _Registration] | None:
    if (
        len(call.args) != 1
        or {kw.arg for kw in call.keywords}
        != {
            "interfaces",
            "tests",
        }
        or len(call.keywords) != 2
    ):
        return None
    try:
        name = ast.literal_eval(call.args[0])
        arguments = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None
    if not isinstance(name, str):
        return None
    interfaces = arguments["interfaces"]
    tests = arguments["tests"]
    if not all(
        isinstance(values, (tuple, list))
        and values
        and all(isinstance(value, str) for value in values)
        for values in (interfaces, tests)
    ):
        return None
    if any(
        Path(test).is_absolute()
        or ".." in Path(test).parts
        or Path(test).parts[:1] != ("tests",)
        or not Path(test).name.startswith("test_")
        or Path(test).suffix != ".py"
        for test in tests
    ):
        return None
    return name, _Registration(tuple(interfaces), tuple(tests))


def _is_policy_assignment(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "TEST_SELECTION_POLICY"
    )


def _valid_policy(value: ast.expr) -> bool:
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "SelectionPolicy"
        and not value.args
        and len({kw.arg for kw in value.keywords}) == len(value.keywords)
        and all(
            kw.arg in {"ignored_paths", "forced_full_paths"} for kw in value.keywords
        )
    ):
        return False
    try:
        patterns = [ast.literal_eval(kw.value) for kw in value.keywords]
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False
    return all(
        isinstance(entries, (tuple, list))
        and all(isinstance(pattern, str) for pattern in entries)
        for entries in patterns
    )
