# Verification Summary

## Scope

The current verification package checks the first 20 MNIST test samples for two checkpoint seeds:

```text
seed1234
seed2026
```

## Compared Outputs

The SystemVerilog testbench compares:

```text
PL logits        vs. Python-generated fixed-point logits
PL pred_class    vs. Python-generated categorical pred_class
```

## Testbench Output Fields

The important result fields are:

```text
checked_samples
pred_mismatch_count
overall_logits_max_abs_diff
```

## Notes

This folder is reserved for copied Questa logs, screenshots, and final result summaries after simulation is rerun for the packaged repository.
