# Scripts

## What This Step Does

Keep helper scripts that support the RTL or packaging flow but are not part of the main inference/export/testbench sequence.

## How To Do It

Current helper:

```text
make_tanh_lut_from_torch.py
```

Run from the repository root:

```text
python scripts/make_tanh_lut_from_torch.py
```

This script generates:

```text
03_pl_golden_rtl/tanh_lut_q412_full.vh
```

Expected role:

```text
utility scripts -> generated lookup tables or packaging support files
```
