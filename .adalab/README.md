# AdaLab deployment

The results-matrix app (`web/`) is mirrored to AdaLab at
<https://adalab.dhigroup.com/apps/tsforecast-benchmark>, so people without a GitHub
account can reach it. GitHub Pages publishes the same app independently via
`.github/workflows/pages.yml`; neither target replaces the other.

Deploys are manual. This repository is public, and GitHub does not let a public repository
call the deploy workflow in the internal `DHI-Internal/dhi_adalab` repo or use its
self-hosted runner. So nothing deploys on a push to `main`: **after publishing new results,
redeploy by hand**, or the AdaLab copy keeps serving the old ones.

Nothing is built locally. AdaLab's own build server clones this repo at the given commit
and builds `../Containerfile` itself.

## Deploying

From the repository root, on the DHI network, with the `dhi-adalab` CLI installed and an
AdaLab personal access token:

```bash
git switch main && git pull
SHA=$(git rev-parse HEAD)
ADALAB_PAT=<token> dhi-adalab app deploy \
  --git-url https://github.com/DHI/timeseries-forecasting-benchmark \
  --commit "$SHA" \
  --spec-file .adalab/app.json \
  --next-image-version "tsforecast-benchmark-web=$SHA"
```

Tag the image with the commit SHA, as above. AdaLab tags cannot change, so redeploying with
a tag it already has silently serves the old image.

The deploy writes `current_image_version` and `next_image_version` back into
`local_container_1.json`. Commit that change on a branch and open a PR, so the files keep
matching what is live.

## Three things that will bite

**`access_level` applies at creation only.** dhi-adalab reads it when it first creates
the app; on later runs it repoints containers and restarts, and never touches the ACL.
Editing `access_level` here after the first deploy changes nothing on its own — change it
in the AdaLab UI, or call `adalib.apps.edit_app(app_id=..., acl_type=..., acl_list=[])`.
The value kept here is the intent, and what a fresh deploy would apply.

**The path prefix must be baked into the build.** AdaLab strips `/apps/<app_url>` from
requests arriving at the container, but the browser still asks for the URLs written into
`index.html`. A bare `/assets/...` resolves against the AdaLab root and never reaches the
app, which renders as a blank page. `APP_BASE` in `../Containerfile` carries the prefix
and has to stay in step with `app_url` below.

**The Gallery card exists but is not tracked here.** It was created by hand in AdaLab, so
there is deliberately no `card.json` in this directory. Adding one later with the default
`card_id: null` makes dhi-adalab decide "create" and produce a *second* card pointing at
the same app. Fetch the existing card's id first (`adalib.cards`) and put it in `card_id`
before committing any `card.json`, then pass `--card .adalab/card.json` to the deploy
command.

## Files

- `app.json` — app identity, URL slug, ACL. `app_id` points the deploy at the live app;
  never edit it. Without it, a deploy creates a second app.
- `local_container_1.json` — image name, port, resources. `metadata_id` and the image
  versions are written back by each deploy.
