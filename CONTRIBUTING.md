# Contributing to RadarSim

Thank you for your interest in contributing to RadarSim! This document provides guidelines for contributions.

## How to Contribute

### 🐛 Reporting Bugs

1. Search existing issues to avoid duplicates
2. Use the bug report template
3. Include:
   - Python version and OS
   - Steps to reproduce
   - Expected vs actual behavior
   - Error traceback if applicable

### 💡 Feature Requests

1. Check existing issues and roadmap
2. Describe the feature and use case
3. Explain why it benefits the project

### 🔧 Pull Requests

1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/your-feature-name`
3. **Make changes** following code style
4. **Test**: Run `pytest tests/` before submitting
5. **Commit**: Use descriptive commit messages
6. **Push**: `git push origin feature/your-feature-name`
7. **Open PR**: Reference related issues

## Code Style

- Follow PEP 8
- Use type hints for function signatures
- Add docstrings with NumPy style
- Keep functions focused and testable

## Physics Contributions

RadarSim values scientific accuracy. When adding physics-related code:

1. **Cite sources** in docstrings (IEEE, ITU, textbooks)
2. **Validate** against published results when possible
3. **Document units** clearly (SI preferred)
4. **Use Numba JIT** for performance-critical loops

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_physics_core.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Questions? Open a discussion or reach out to the maintainers!
