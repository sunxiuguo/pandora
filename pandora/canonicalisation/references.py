from __future__ import annotations

from pandora.canonicalisation.assemblies import _spans_within
from pandora.schemas.structure import AssemblyRecord, Structure


def _prune_assembly_chains(
    assembly: AssemblyRecord, kept: set[str]
) -> AssemblyRecord:
    """Keep assembly metadata and operators, pruning chain instructions."""
    generators = []
    for generator in assembly.generators:
        chains = [c for c in generator.asym_id_list if c in kept]
        if chains:
            generators.append(
                generator.model_copy(update={"asym_id_list": chains})
            )
    return assembly.model_copy(update={"generators": generators})


def _prune_chain_references(structure: Structure) -> Structure:
    """Remove typed references to absent asym units and orphan entities."""
    kept = {u.id for u in structure.asym_units}
    entity_ids = {u.entity_id for u in structure.asym_units}
    ss = structure.secondary_structure
    return structure.model_copy(
        update={
            "entities": [e for e in structure.entities if e.id in entity_ids],
            "connections": [
                c
                for c in structure.connections
                if c.ptnr1.label_asym_id in kept
                and c.ptnr2.label_asym_id in kept
            ],
            "secondary_structure": ss.model_copy(
                update={
                    "conf_records": _spans_within(ss.conf_records, kept),
                    "sheet_strands": _spans_within(ss.sheet_strands, kept),
                }
            ),
            "assemblies": [
                _prune_assembly_chains(a, kept) for a in structure.assemblies
            ],
        }
    )
