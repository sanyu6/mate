# S6000 baseline snapshot

This branch preserves the MATE operator sources from snapshot
`f9e72b8214a5af0f2c2b5d2b9bd6ff7ec487246a`, paired with SGLang
`4174b21a1193321309c518d3a657f88ed191df5a`.

The FlashMLA wrapper is included under `wrappers/FlashMLA`. The complete
MUTLASS include tree is vendored under `3rdparty/mutlass/include`, including
the W4A4 pool TME cache-hint patch. It must not be replaced with an unpatched
MUTLASS submodule when reproducing this snapshot.

The operator sources and MUTLASS headers are byte-identical to the source
snapshot. Publication removes private development material and deployment
records. Dependency versions, runtime configuration, and model artifacts
must still match the intended workload; this publication does not constitute
a fresh performance or accuracy validation.
