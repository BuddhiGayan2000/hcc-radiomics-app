# data/

Holds sample inputs and known-good reference values used for testing — not part of the application itself.

```
data/
├── test_images/          Sample post/pre-contrast images for manual smoke-testing the upload → ROI → predict flow
└── expected_features/    Known feature values (from the original Python extraction) for parity testing
```

**Currently empty.** Neither folder was populated as part of initial setup, because the source material needed to fill them wasn't part of the delivered files:
- `test_images/` needs real (or realistic) CT slice pairs — none were provided; the WhatsApp screenshots kept alongside the original deliverables are notebook output plots, not usable CT images.
- `expected_features/` needs the original Python-computed feature values for specific slices (e.g. an exported feature CSV) — this exists somewhere in the original research project but was not part of what was handed off for this application build.

**Get these from the research team before running the parity test** — see [../docs/PARITY_TESTING.md](../docs/PARITY_TESTING.md). Until then, use any two arbitrary images for UI smoke-testing (the pipeline doesn't care what the images actually show — it just needs pixels), but treat any resulting prediction as structurally-working, not clinically meaningful.
