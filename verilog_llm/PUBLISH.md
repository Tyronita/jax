# Publishing this as its own repository

This project currently lives inside another repo's branch. It is fully
self-contained (its own `README`, `LICENSE`, `.gitignore`, `docs/`), so you can
lift it into a standalone GitHub repo in one of two ways.

> Heads up: automated tooling here didn't have permission to create a new repo
> under your account, so the final "create + push" is a manual step (below).
> The `publish.sh` script automates everything except the GitHub-side repo
> creation.

## Option A — with the GitHub CLI (one command)

From inside the `verilog_llm/` directory:

```bash
./publish.sh tiny-gpt-verilog --public      # or --private
```

This will: init a fresh git repo from this directory's contents, make a clean
initial commit, create the GitHub repo via `gh`, and push `main`. Requires
`gh auth login` to have been run once.

## Option B — manual (no gh)

1. Create an empty repo on GitHub named `tiny-gpt-verilog` (no README/license —
   we provide our own).
2. From inside `verilog_llm/`:

```bash
git init -b main
git add .
git commit -m "Tiny GPT-style LLM in Verilog"
git remote add origin git@github.com:<you>/tiny-gpt-verilog.git
git push -u origin main
```

That's it — the directory becomes the root of the new repo.
