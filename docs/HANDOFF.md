# Handoff: continue fin-news-sentiment-lab on another machine

State on 2026-11-15: the latest commit on branch `claude/project-skeleton-build-dj1aj6` is
`bdeec3a`. That branch is the only one on GitHub (there is no `main` yet). CI is green.

The prompt below is for Claude Code on the new machine. Open a terminal in an empty folder
where you want the project, start `claude`, and paste everything between the lines.

---

```text
I am continuing my project fin-news-sentiment-lab on a new machine. Set it up and run
every test. Do not change, train or regenerate anything yet.

Repository: https://github.com/samarthkala5/fin-news-sentiment-lab
Branch: claude/project-skeleton-build-dj1aj6 (the only branch; `main` does not exist yet).
Last known commit: bdeec3a "add push-to-main step to guide".

Steps:
1. Clone the repository and check out the branch above. If the repository is private
   and the clone fails, stop and tell me how to authenticate. Do not put a token in any
   file. Confirm that HEAD is bdeec3a or a later commit on that branch.
2. Read CLAUDE.md, PROJECT_PLAN.md and docs/PROJECT_GUIDE.md (sections 1-4). They hold
   the rules and the current state. Follow CLAUDE.md for the whole session.
3. Check that python3 is 3.11, 3.12 or 3.13 (pyproject requires >= 3.11; CI uses 3.11).
   If it is not, tell me and stop.
4. Create a virtual environment in .venv inside the repository and install, in order:
     pip install -r requirements.txt
     pip install -e . --no-deps
     pip install "torch>=2.4,<2.15" --index-url https://download.pytorch.org/whl/cpu
     pip install -r requirements-local.txt
   On macOS, if the CPU index has no wheel for this machine, install
   "torch>=2.4,<2.15" from PyPI instead. Do not install requirements-gpu.txt; Unsloth
   needs an NVIDIA GPU and runs only in Colab.
5. Run, from the repository root with the venv active:
     ruff check .
     pytest -m "not heavy"
     pytest -m heavy
     pytest --collect-only -q | tail -1
   The heavy tests download small public models from the Hugging Face Hub on the first
   run (MiniLM, the DistilBERT tokenizer, SmolLM2-135M), so they need internet access.
6. Report back:
   - the commit you are on and the Python version;
   - each command's result with pass, fail and skip counts;
   - the full error output for anything that failed, with your diagnosis.
   Expected on the last machine: ruff clean; 199 passed in "not heavy"; 37 passed in
   "heavy"; 236 tests collected. A difference is not automatically a bug (for example a
   platform-specific skip), but explain every one.
7. Then run `git status`. It must show no changes: tests must not write into results/,
   README.md or data/. If anything changed, show the diff and do not commit it.

Rules that matter from now on (also in CLAUDE.md):
- Never invent numbers. Results come only from results/metrics/*.json written by the
  scripts. Only the TF-IDF baseline, K-Means, the comparison file and the RAG index have
  real outputs; Keras, DistilBERT, the LLM and the RAG evaluation have had smoke runs only.
- data/splits/ is read-only. The test split is for final evaluation only.
- Run `pytest -m "not heavy"` and `ruff check .` before every commit; commit messages
  are "<verb> <what>". Work on one plan day at a time.
- Next steps are in docs/PROJECT_GUIDE.md section 3. Step 1 there (create `main`, make
  the repo public, switch to a local `main`) is mine to do; ask me before pushing to
  GitHub.

When the report is done, stop and wait for me.
```

---

## Notes for you

- If you already have Claude Code with GitHub access on the new machine, it can clone the
  repository directly. Otherwise run `git clone` yourself first, then start `claude`
  inside the folder and drop step 1.
- After `main` exists (guide section 3, step 1), change "Branch:" in the prompt to `main`.
- The documentation is in `docs/PROJECT_GUIDE.md` and `docs/math/`. There is also a
  rendered copy, private to your claude.ai account: open it from your artifacts as
  "Sentiment Lab Guide".
