# Security policy

## Supported releases

Security fixes are provided for the latest major release. Version 3.x is currently supported; older releases should be upgraded before reporting a version-specific issue.

## Reporting a vulnerability

Do not open a public issue containing an exploitable vulnerability. Use the repository’s private GitHub Security Advisory reporting channel: `Security` → `Advisories` → `Report a vulnerability`. Include affected versions, operating system, reproduction steps, impact, and a minimal proof of concept. The maintainer will acknowledge a complete report as soon as practical and coordinate disclosure after a fix is available.

## Security boundaries

RadarSim does not require an account, telemetry service, remote API, or network connection at runtime. It reads local scenario and recording files and may export local files selected by the user.

YAML uses `safe_load`, and replay metadata uses JSON parsing. These choices prevent intentional execution of serialized Python objects, but untrusted files can still consume memory or CPU. Open scenarios and HDF5 recordings only from trusted sources, keep scientific Python and PySide6 dependencies updated, and run third-party extensions in an isolated environment.

Desktop release artifacts are built by the public GitHub Actions workflow. Verify the release tag and repository origin before running a downloaded executable.
