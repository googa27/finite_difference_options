# Compiled PDE fixtures

Public-synthetic serialized compiled `pde_ir.v0` artifacts consumed by the FD issue #141 adapter.

`black_scholes_call_v0.json` wraps the FPF public Black-Scholes `pde_ir.v0` source fixture and its restricted symbolic compiler output. The FD adapter treats this JSON as a public contract fixture, never as a source-tree import, and solves only when the source IR hash, compiled-operator hash, units, measure, numeraire, time orientation, boundary semantics and solver controls match exactly.
