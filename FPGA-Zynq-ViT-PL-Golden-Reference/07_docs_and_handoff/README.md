# 07 Docs And Handoff

## Purpose

This step contains project-level documentation for the design boundary, verification method, and engineering status.

## Files

```text
README_PL_BOUNDARY.md
```

Explains the Zynq-style PS/PL inference boundary: Python/PS prepares `embedding_output`, and PL computes the encoder, norms, classifier, logits, and `pred_class`.

```text
FINAL_VIT_PL_GOLDEN_README_EN.md
```

Long-form handoff document that describes the repository flow, numeric format, key RTL files, payload locations, reproduction commands, and current engineering status.

## Main Topics

```text
PS/PL split
embedding_output boundary
Q4.12 fixed-point format
classifier-in-PL decision
20-sample verification flow
current limits
next architecture step
```

## Result Of This Step

```text
implementation + verification flow -> clear handoff for GitHub reviewers
```
