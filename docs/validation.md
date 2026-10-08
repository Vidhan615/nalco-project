# Validation record

Checked on 8 October 2026 using a fresh Python 3.11.9 environment with the pinned
dependencies in `requirements-dev.txt`.

- **20 tests passed.** Hand-calculated cases cover next-open execution, terminal
  liquidation and fees, long/short cash accounting, month-boundary attribution,
  target resets, multi-asset rebalancing, retained Supertrend bands and crossings,
  causal feature prefixes, strictly earlier external inputs, label-boundary
  separation, training-only scaling, seeded training and saved-policy restoration.
- **Synthetic smoke run passed** after installing dependencies in a new local
  environment. The generated demo uses fictional data and is labelled as such.
- **12 corrected historical simulations reconciled.** Terminal cash agrees
  independently with initial cash plus every signed fill minus fees. Monthly
  return products match total portfolio returns. Intraday positions close within
  the same execution session, and all signal decisions precede their fills.
- **Four annual baseline windows and four cost scenarios generated.** Annual
  baseline training expands chronologically. Cost scenarios reuse seed 42's
  base-cost policy; changed portfolio observations can change its trades.
- **Publication file list checked.** Raw price files, virtual environments,
  CV/other PDFs, credentials, local caches and uncompressed model duplicates
  are excluded. Selected original audit tables are labelled separately.

The workflow in `.github/workflows/ci.yml` repeats tests and a synthetic smoke
run on Linux/Python 3.11 and 3.12. Its live status is reported by GitHub Actions;
local results do not imply a remote CI run has completed.

These checks establish internal software/accounting behavior under the documented
assumptions. They do not certify source-price provenance, a fully realistic
exchange/broker simulation, statistical significance or profitable live trading.
