# Original implementation audit

These tables were produced by the original 5 October 2026 audit. They describe
the original supplied data, models and saved simulations, not the corrected
package. The full audit narrative is in
[docs/legacy/original-project-analysis.md](../../docs/legacy/original-project-analysis.md).

Corrected simulator outputs are separately stored under `results/historical/`.
Differences in targets, feature definitions, native-market return alignment,
test-session boundaries and costs mean old and new metrics are not interchangeable.
In particular, the old forecast table includes next-close and multi-session
targets; the corrected baseline focuses on next-session open-to-close returns.

Original scripts/raw data/reports are preserved in the owner's source workspace;
this curated repository does not redistribute the raw input files or legacy PDFs.
