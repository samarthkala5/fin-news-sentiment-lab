# CLAUDE.md
Project: fin-news-sentiment-lab. Full spec in PROJECT_PLAN.md.

Rules
- Work on one day of the plan at a time. At the end, report what was built, the test results, and the commit hash, then wait.
- Never invent metrics. All numbers come from results/metrics/*.json written by scripts.
- Do not modify files outside this repo. Do not touch my other projects.
- Check current library docs/versions before using LangChain, Unsloth, TensorFlow, datasets, or transformers APIs.
- Every training script needs a --smoke flag (tiny data, 1 epoch, CPU).
- Fixed splits in data/splits/ are read-only after Day 1. Test set is for final evaluation only.
- Keep code simple and commented; I must be able to explain it.
- Run `pytest -m "not heavy"` and `ruff check .` before every commit.
- GPU work goes through notebooks/colab_runner.ipynb; logic lives in scripts/ and src/.
- One focused commit per day, message "<verb> <what>".
