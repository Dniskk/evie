# evie Documentation

Welcome to the evie documentation! This directory contains all our learnings, decisions, and architectural documentation for the LLM training framework.

## Structure

### 📐 [architecture/](architecture/)
Technical architecture and system design documentation.

- **model-design.md** - Transformer architecture details and design choices
- **tokenization.md** - Tokenizer design and implementation
- **training-pipeline.md** - Training system architecture and data flow

### 🎯 [decisions/](decisions/)
Architectural Decision Records (ADRs) documenting important choices.

- **template.md** - Template for new ADRs
- **001-dataset-choice.md** - Dataset selection rationale
- **002-optimizer.md** - Optimizer and hyperparameter choices

Each ADR follows the format: Context → Decision → Consequences

### 💡 [learnings/](learnings/)
Knowledge gained during development and training.

- **scaling-laws.md** - Insights about model scaling
- **hyperparameters.md** - Hyperparameter tuning discoveries
- **debugging.md** - Common issues and their solutions

### 🚀 [training/](training/)
Training strategy, datasets, and evaluation.

- **strategy.md** - Overall training approach and goals
- **datasets.md** - Dataset details and preprocessing
- **evaluation.md** - Metrics and evaluation methodology

## Contributing to Docs

When you learn something important or make a significant decision:

1. **Learnings**: Add to the appropriate file in `learnings/` or create a new one
2. **Decisions**: Create a new ADR in `decisions/` using the template
3. **Architecture**: Update relevant architecture docs when design changes
4. **Training**: Document training runs, results, and insights

Keep files:
- Focused on a single topic
- Under 500 lines (split if needed)
- Cross-referenced with related docs
- Updated as we learn and evolve

## Quick Links

- [Project README](../README.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
