# AdaLab deployment

The results-matrix app (`web/`) is mirrored to AdaLab at
<https://adalab.dhigroup.com/apps/tsforecast-benchmark>, so people without a GitHub
account can reach it. GitHub Pages publishes the same app independently via
`.github/workflows/pages.yml`; neither target replaces the other.

`.github/workflows/adalab.yml` deploys on every push to `main` that touches the app.
Nothing is built in CI — AdaLab's own build server clones this repo at the pushed commit
and builds `../Containerfile` itself, which is why no Harbor credentials are needed. The
only secret involved is `ADALAB_PAT`.

## Two things that will bite

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

## Files

- `app.json` — app identity, URL slug, ACL. `app_id` is written back by CI; never edit it.
- `local_container_1.json` — image name, port, resources. `metadata_id` and
  `current_image_version` are written back by CI. `next_image_version` is overridden per
  run with the commit SHA, so it does not need bumping by hand.
