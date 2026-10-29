from __future__ import annotations

from pathlib import Path

import pytest

from pandora.canonicalisation import canonicalise_structure
from pandora.datasets import curate_structure
from pandora.export import structure_to_mmcif
from pandora.parsing import mmcif_to_structure
from pandora.schemas.canonicalisation import canonicalisationPolicy
from pandora.schemas.dataset import ContentRules, DatasetCurationPolicy
from pandora.schemas.structure import Structure

MMCIF_DIR = Path(__file__).parent.parent / "datasets" / "dev" / "mmcif"
CHAINS = ["keep", "twin", "drop", "ligand", "water", "ion", "branch"]
POLICY = dict(policy_id="refs", policy_name="refs", policy_version="1")


def _structure() -> Structure:
    """Small complete and gapped polymers plus every content-filter class."""
    entities = [
        {"id": "1", "type": "polymer", "pdbx_description": "retained protein"},
        {"id": "2", "type": "polymer"},
        {"id": "3", "type": "non-polymer", "pdbx_description": "ligand"},
        {"id": "4", "type": "water"},
        {"id": "5", "type": "non-polymer", "pdbx_description": "ZINC ION"},
        {"id": "6", "type": "branched"},
    ]
    entity_ids = ["1", "1", "2", "3", "4", "5", "6"]
    atoms = []
    for chain, entity_id in zip(CHAINS, entity_ids):
        polymer = chain in CHAINS[:3]
        for seq in [1, 3] if chain == "drop" else [1]:
            names = (
                ["CA"]
                if chain == "drop" or not polymer
                else ["N", "CA", "C", "O"]
            )
            for name in names:
                atoms.append(
                    dict(
                        id=len(atoms) + 1,
                        group_PDB="ATOM" if polymer else "HETATM",
                        type_symbol="C",
                        label_atom_id=name,
                        label_comp_id="ALA" if polymer else "LIG",
                        label_asym_id=chain,
                        label_entity_id=entity_id,
                        label_seq_id=seq if polymer else None,
                        auth_seq_id=str(seq),
                        auth_asym_id=f"auth_{chain}",
                        Cartn_x=0.0,
                        Cartn_y=0.0,
                        Cartn_z=0.0,
                        occupancy=1.0,
                        B_iso_or_equiv=20.0,
                    )
                )
    pairs = [("keep", "twin")]
    for chain in CHAINS[2:]:
        pairs.extend([("keep", chain), (chain, "keep")])
    spans = [
        dict(
            id=str(i),
            beg_label_asym_id=a,
            end_label_asym_id=b,
            beg_label_seq_id=1,
            end_label_seq_id=1,
        )
        for i, (a, b) in enumerate(pairs)
    ]
    return Structure.model_validate(
        dict(
            entry_id="refs",
            entry={"id": "refs", "title": "preserve metadata"},
            entities=entities,
            asym_units=[
                dict(id=c, entity_id=e, auth_id=f"auth_{c}")
                for c, e in zip(CHAINS, entity_ids)
            ],
            atoms=atoms,
            connections=[
                dict(
                    id=str(i),
                    conn_type_id="covale",
                    details="deposited",
                    ptnr1=dict(label_asym_id=a, label_comp_id="ALA"),
                    ptnr2=dict(label_asym_id=b, label_comp_id="ALA"),
                )
                for i, (a, b) in enumerate(pairs)
            ],
            secondary_structure=dict(
                conf_records=[
                    dict(**span, conf_type_id="HELX_P") for span in spans
                ],
                sheet_strands=[
                    dict(**span, sheet_id="sheet") for span in spans
                ],
            ),
            assemblies=[
                dict(
                    id="1",
                    details="deposited",
                    method_details="author",
                    oligomeric_details="dimeric",
                    oligomeric_count=2,
                    author_determined=True,
                    operators=[
                        dict(
                            id="1",
                            type="identity operation",
                            matrix=[
                                [1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0],
                            ],
                            vector=[0.0, 0.0, 0.0],
                        )
                    ],
                    generators=[
                        dict(
                            assembly_id="1",
                            oper_expression="1",
                            asym_id_list=ids,
                        )
                        for ids in [CHAINS, ["ligand"], ["drop"]]
                    ],
                ),
                dict(
                    id="concrete",
                    details="already materialized",
                    oligomeric_count=7,
                ),
            ],
            raw={
                "_chain_note": [{"label_asym_id": "keep", "note": "indexed"}],
                "_unrelated": [{"note": "verbatim"}],
            },
        )
    )


def _assert_references(s: Structure) -> None:
    """Check each typed chain reference and entity ownership."""
    kept = {u.id for u in s.asym_units}
    assert {a.label_asym_id for a in s.atoms} <= kept
    assert {e.id for e in s.entities} == {u.entity_id for u in s.asym_units}
    assert all(
        set(g.asym_id_list) <= kept for a in s.assemblies for g in a.generators
    )
    assert all(
        c.ptnr1.label_asym_id in kept and c.ptnr2.label_asym_id in kept
        for c in s.connections
    )
    assert all(
        r.beg_label_asym_id in kept and r.end_label_asym_id in kept
        for r in [
            *s.secondary_structure.conf_records,
            *s.secondary_structure.sheet_strands,
        ]
    )


@pytest.mark.parametrize(
    "chain_strategy", ["preserve", "remap", "use_auth_chain_id"]
)
@pytest.mark.parametrize(
    "mode",
    [
        "filter",
        "annotate_only",
        "missing_atoms",
        "missing_residues",
        "incomplete_chains",
        "content",
        "chain_rules",
    ],
)
def test_chain_removal_prunes_all_typed_references(
    mode: str, chain_strategy: str
) -> None:
    original = _structure()
    before = original.model_dump()
    content = mode in {"filter", "annotate_only", "content"}
    kept = CHAINS[:3] if content else [c for c in CHAINS if c != "drop"]
    overrides = {}
    if mode in {"filter", "annotate_only"}:
        overrides["ligand_rules"] = dict(
            strategy=mode,
            keep_waters=False,
            keep_ions=False,
            keep_nonpolymer_ligands=False,
        )
    elif mode in {"missing_atoms", "missing_residues", "incomplete_chains"}:
        strategy = {
            "missing_atoms": "drop_partial_residue",
            "missing_residues": "drop_chain_segment",
            "incomplete_chains": "exclude",
        }[mode]
        overrides["missing_data_rules"] = {mode: {"strategy": strategy}}
    if mode in {"content", "chain_rules"}:
        # Exercise curation with the same remapped input as canonicalisation.
        source = canonicalise_structure(
            original,
            canonicalisationPolicy(
                **POLICY,
                identifier_rules={"chain_id": {"strategy": chain_strategy}},
            ),
        )[0]
        rules = (
            dict(
                content_rules=ContentRules(
                    keep_ligands=False, keep_waters=False, keep_ions=False
                )
            )
            if content
            else dict(quality_rules={"max_chain_length": 1})
        )
        result, exclusions, provenance = curate_structure(
            source, None, DatasetCurationPolicy(**POLICY, **rules)
        )
        assert result is not None
        assert [e.reason_code for e in exclusions] == (
            [] if content else ["CHAIN_TOO_LONG"]
        )
    else:
        result, _, provenance = canonicalise_structure(
            original,
            canonicalisationPolicy(
                **POLICY,
                identifier_rules={"chain_id": {"strategy": chain_strategy}},
                **overrides,
            ),
        )
    chain_map = dict(
        zip(
            CHAINS,
            CHAINS
            if chain_strategy == "preserve"
            else [chr(65 + i) for i in range(len(CHAINS))]
            if chain_strategy == "remap"
            else [f"auth_{c}" for c in CHAINS],
        )
    )
    assert {u.id for u in result.asym_units} == {chain_map[c] for c in kept}
    assert [a.id for a in result.atoms] == [
        a.id for a in original.atoms if a.label_asym_id in kept
    ]
    expected_ids = [
        c.id
        for c in original.connections
        if c.ptnr1.label_asym_id in kept and c.ptnr2.label_asym_id in kept
    ]
    assert [c.id for c in result.connections] == expected_ids
    for old, new in zip(
        [c for c in original.connections if c.id in expected_ids],
        result.connections,
    ):
        expected = old.model_dump()
        for partner in ("ptnr1", "ptnr2"):
            expected[partner]["label_asym_id"] = chain_map[
                expected[partner]["label_asym_id"]
            ]
        assert new.model_dump() == expected
    assert result.entities == [
        e
        for e in original.entities
        if e.id in {u.entity_id for u in result.asym_units}
    ]
    assert [
        r.id for r in result.secondary_structure.conf_records
    ] == expected_ids
    assert [
        r.id for r in result.secondary_structure.sheet_strands
    ] == expected_ids
    for old, new in zip(original.assemblies, result.assemblies):
        assert new.model_dump(exclude={"generators"}) == old.model_dump(
            exclude={"generators"}
        )
    assert [g.asym_id_list for g in result.assemblies[0].generators] == [
        [chain_map[c] for c in g.asym_id_list if c in kept]
        for g in original.assemblies[0].generators
        if set(g.asym_id_list) & set(kept)
    ]
    assert result.entry == original.entry
    assert result.raw == {"_unrelated": [{"note": "verbatim"}]}
    assert "_chain_note" in provenance.dropped_raw_categories or (
        mode in {"content", "chain_rules"} and chain_strategy != "preserve"
    )
    _assert_references(result)
    assert original.model_dump() == before


def test_default_policies_preserve_complete_structure() -> None:
    source = _structure()
    canonical = canonicalise_structure(
        source, canonicalisationPolicy(**POLICY)
    )[0]
    curated, exclusions, _ = curate_structure(
        source, None, DatasetCurationPolicy(**POLICY)
    )
    assert canonical == source
    assert curated == source
    assert exclusions == []


@pytest.mark.parametrize(
    "keep_ligands,keep_waters,keep_ions",
    [
        (True, True, True),
        (False, True, True),
        (True, False, True),
        (True, True, False),
        (False, False, False),
    ],
)
def test_partial_content_filter_keeps_requested_classes(
    keep_ligands: bool, keep_waters: bool, keep_ions: bool
) -> None:
    result, _, _ = curate_structure(
        _structure(),
        None,
        DatasetCurationPolicy(
            **POLICY,
            content_rules=ContentRules(
                keep_ligands=keep_ligands,
                keep_waters=keep_waters,
                keep_ions=keep_ions,
            ),
        ),
    )
    assert result is not None
    expected = (
        set(CHAINS[:3])
        | ({"ligand", "branch"} if keep_ligands else set())
        | ({"water"} if keep_waters else set())
        | ({"ion"} if keep_ions else set())
    )
    assert {u.id for u in result.asym_units} == expected
    _assert_references(result)


@pytest.mark.parametrize(
    "mode,entry",
    [("canonical", "1ayi"), ("canonical", "104m"), ("curation", "104m")],
)
def test_real_fixture_filter_export_reparse(
    mode: str, entry: str, tmp_path: Path
) -> None:
    source = mmcif_to_structure(str(MMCIF_DIR / entry[1:3] / f"{entry}.cif"))[0]
    before = source.model_dump()
    if mode == "canonical":
        filtered = canonicalise_structure(
            source,
            canonicalisationPolicy(
                **POLICY,
                ligand_rules={
                    "strategy": "filter",
                    "keep_waters": False,
                    "keep_ions": False,
                    "keep_nonpolymer_ligands": False,
                },
            ),
        )[0]
    else:
        filtered, exclusions, _ = curate_structure(
            source,
            None,
            DatasetCurationPolicy(
                **POLICY,
                content_rules=ContentRules(
                    keep_ligands=False, keep_waters=False, keep_ions=False
                ),
            ),
        )
        assert exclusions == []
    assert filtered is not None
    assert filtered.atoms
    _assert_references(filtered)
    path = tmp_path / "filtered.cif"
    structure_to_mmcif(filtered, path)
    reparsed = mmcif_to_structure(str(path))[0]
    _assert_references(reparsed)
    assert len(reparsed.atoms) == len(filtered.atoms)
    assert {u.id for u in reparsed.asym_units} == {
        u.id for u in filtered.asym_units
    }
    assert [c.id for c in reparsed.connections] == [
        c.id for c in filtered.connections
    ]
    assert source.model_dump() == before


@pytest.mark.parametrize("content_filter", [False, True])
@pytest.mark.parametrize(
    "kind", ["fixture", "empty", "atomless_polymer", "ligand_only"]
)
def test_curation_excludes_entries_without_polymer_atoms(
    kind: str, content_filter: bool
) -> None:
    if kind == "fixture":
        source = mmcif_to_structure(str(MMCIF_DIR / "2a" / "22ag.cif"))[0]
    else:
        source = _structure()
        atoms = (
            []
            if kind != "ligand_only"
            else [a for a in source.atoms if a.label_asym_id == "ligand"]
        )
        source = source.model_copy(update={"atoms": atoms})
    if kind == "empty":
        source = source.model_copy(update={"asym_units": [], "entities": []})
    before = source.model_dump()
    result, exclusions, provenance = curate_structure(
        source,
        None,
        DatasetCurationPolicy(
            **POLICY,
            content_rules=ContentRules(
                keep_ligands=not content_filter,
                keep_waters=not content_filter,
                keep_ions=not content_filter,
            ),
        ),
    )
    assert result is None
    assert [e.reason_code for e in exclusions] == ["NO_CHAINS_LEFT"]
    assert exclusions[0].chain_id is None
    assert provenance.policy_id == "refs"
    assert source.model_dump() == before


def test_canonical_empty_output_only_prunes_references() -> None:
    source = _structure()
    source = source.model_copy(
        update={
            "atoms": [a for a in source.atoms if a.label_asym_id == "ligand"],
            "asym_units": [u for u in source.asym_units if u.id == "ligand"],
        }
    )
    result = canonicalise_structure(
        source,
        canonicalisationPolicy(
            **POLICY, ligand_rules={"strategy": "annotate_only"}
        ),
    )[0]
    assert (
        result.atoms
        == result.asym_units
        == result.entities
        == result.connections
        == []
    )
    assert (
        result.secondary_structure.conf_records
        == result.secondary_structure.sheet_strands
        == []
    )
    assert len(result.assemblies) == len(source.assemblies)
    assert all(a.generators == [] for a in result.assemblies)


@pytest.mark.parametrize("content_filter", [False, True])
def test_curation_counts_polymer_hetatm_atoms(content_filter: bool) -> None:
    source = _structure()
    source = source.model_copy(
        update={
            "atoms": [
                a.model_copy(update={"group_PDB": "HETATM"})
                for a in source.atoms
            ]
        }
    )
    result, exclusions, _ = curate_structure(
        source,
        None,
        DatasetCurationPolicy(
            **POLICY,
            content_rules=dict(
                keep_ligands=not content_filter,
                keep_waters=not content_filter,
                keep_ions=not content_filter,
            ),
        ),
    )
    assert result is not None
    assert exclusions == []
    assert {a.label_asym_id for a in result.atoms} >= set(CHAINS[:3])


def test_shared_entity_survives_removal_of_one_chain() -> None:
    source = _structure()
    source = source.model_copy(
        update={
            "asym_units": [
                u.model_copy(update={"entity_id": "1"}) if u.id == "drop" else u
                for u in source.asym_units
            ],
            "atoms": [
                a.model_copy(update={"label_entity_id": "1"})
                if a.label_asym_id == "drop"
                else a
                for a in source.atoms
            ],
            "entities": [e for e in source.entities if e.id != "2"],
        }
    )
    result, exclusions, _ = curate_structure(
        source,
        None,
        DatasetCurationPolicy(**POLICY, quality_rules={"max_chain_length": 1}),
    )
    assert result is not None
    assert [e.chain_id for e in exclusions] == ["drop"]
    assert result.entities == source.entities
    _assert_references(result)


@pytest.mark.parametrize(
    "chain_strategy", ["preserve", "remap", "use_auth_chain_id"]
)
def test_filtered_real_assembly_can_be_materialized(
    chain_strategy: str, tmp_path: Path
) -> None:
    source = mmcif_to_structure(str(MMCIF_DIR / "04" / "104m.cif"))[0]
    filtered = canonicalise_structure(
        source,
        canonicalisationPolicy(
            **POLICY,
            identifier_rules={"chain_id": {"strategy": chain_strategy}},
            ligand_rules=dict(
                strategy="filter",
                keep_waters=False,
                keep_ions=False,
                keep_nonpolymer_ligands=False,
            ),
        ),
    )[0]
    assert filtered.assemblies[0].generators
    expanded = canonicalise_structure(
        filtered,
        canonicalisationPolicy(
            **POLICY,
            assembly_rules={"strategy": "standardize_biological_assembly"},
        ),
    )[0]
    assert expanded.atoms
    _assert_references(expanded)
    assert len(expanded.assemblies) == 1
    assert expanded.assemblies[0].generators == []
    assert (
        expanded.assemblies[0].oligomeric_count
        == filtered.assemblies[0].oligomeric_count
    )
    # A later filter preserves the legitimate concrete assembly record.
    curated, _, _ = curate_structure(
        expanded,
        None,
        DatasetCurationPolicy(
            **POLICY,
            content_rules=dict(
                keep_ligands=False, keep_waters=False, keep_ions=False
            ),
        ),
    )
    assert curated is not None
    assert curated.assemblies == expanded.assemblies
    path = tmp_path / "expanded.cif"
    structure_to_mmcif(curated, path)
    reparsed = mmcif_to_structure(str(path))[0]
    assert len(reparsed.atoms) == len(expanded.atoms)
    _assert_references(reparsed)
