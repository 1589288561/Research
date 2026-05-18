# 06 Verification Results

## Purpose

This step stores reviewer-facing verification evidence and result summaries.

## Files

```text
VERIFICATION_SUMMARY.md
```

Summarizes the verification scope, compared outputs, and important testbench fields.

## How To Produce Results

Run the Questa testbench from:

```text
../05_systemverilog_testbench/
```

Then record the important output fields here:

```text
checked_samples
pred_mismatch_count
overall_logits_max_abs_diff
```

## Current Verification Scope

```text
first 20 MNIST test samples
seed1234
seed2026
```

The testbench compares PL logits and PL `pred_class` against Python-generated fixed-point reference files.

## Result Of This Step

```text
simulation output -> compact evidence that the PL golden model matches the exported reference payload
```
