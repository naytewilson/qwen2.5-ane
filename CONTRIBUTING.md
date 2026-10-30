# Contributing to qwen2.5-ane

This repository is a public research implementation of one Qwen2.5-1.5B ANE
configuration, not a general-purpose benchmark or an Apple-supported SDK.
Useful contributions include reproducible Swift fixes, portability repairs,
documentation corrections and independently checkable measurements.

## Before opening a pull request

- Describe the affected code, platform, model configuration and expected result.
- Keep commits focused. Include a test or concrete before/after validation when possible.
- Run the portable checks: `bash -n setup.sh smoke.sh bench.sh scripts/test-gitignore-symlinks.sh`
  and `bash scripts/test-gitignore-symlinks.sh .`.
- On a suitable macOS host with Xcode 16.3 or later and Swift 6.1 or later,
  run `swift build -c release`. ANE placement and performance require an
  Apple Silicon Mac, a compiled model artifact and the stated workload.
- State clearly when testing was static, a build only, or an actual ANE run.
  A successful CI build does not establish hardware placement, parity or speed.

Do not commit model weights, generated model bundles, private fixtures,
credentials, machine-specific paths, or nonpublic experimental receipts.
Do not assume every upstream converter artifact is redistributable.
Follow the repository's existing license and provenance notices.

For vulnerabilities or accidental disclosures, use the private contact in
[SECURITY.md](SECURITY.md) rather than putting sensitive details in a public issue.
No CLA, academic or institutional affiliation is required.
