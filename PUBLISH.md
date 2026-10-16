# Publish to GitHub

The project is committed locally. GitHub CLI is installed in the development
environment, but its injected `GH_TOKEN` failed authentication, so no remote
repository has been created or overwritten.

Run the following on your own authenticated machine, from this project directory:

```bash
gh auth login --hostname github.com
gh auth status
gh repo create neural-network-from-scratch --public --source=. --remote=origin --push
```

`gh repo create` fails if the repository already exists; do not force-push or
overwrite it. If a stale `GH_TOKEN` is set in your own shell, unset it before
interactive login. Do not paste credentials into source files or commits.
