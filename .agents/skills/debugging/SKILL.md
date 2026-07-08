# Skill: Debugging

## Workflow

```text
Reproduce
→ Isolate
→ Collect evidence
→ One hypothesis
→ Smallest fix
→ Re-run
```

## Environment evidence

- exact command;
- Python version;
- Torch version;
- CUDA availability;
- GPU name;
- Ultralytics version;
- full traceback.

## Video checklist

- path?
- opens?
- frame shape?
- dtype?
- FPS?
- writer?
- output resolution?

## Model checklist

- model loaded?
- device is really CUDA?
- output has detections?
- class filtering correct?
- coordinate mapping correct?

## Do not

- upgrade everything;
- change model;
- rewrite many files;
- hide error with broad try/except.

## Final debug report

```text
Root cause:
Fix:
Why:
Test:
Remaining risk:
```
