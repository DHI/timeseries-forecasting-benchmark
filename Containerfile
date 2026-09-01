# Container image for the results-matrix app (web/), deployed to AdaLab and served at
# https://adalab.dhigroup.com/apps/<app_url>. GitHub Pages is unaffected — it keeps
# building web/ directly via .github/workflows/pages.yml.
#
# AdaLab builds this image server-side from a git URL + commit, so the build context is
# the repository root and everything the app needs must be committed (the results JSON
# under web/public/data/ is).
#
# Two rules from the AdaLab container docs that any edit must keep obeying:
#   - expose exactly one port
#   - define no custom USER / useradd — AdaLab manages user-namespace mappings, and a
#     pinned UID fails with "container doesn't map UID 1000"

# Node 20 to match the Pages workflow, which is where this build is proven.
FROM node:20-alpine AS build

WORKDIR /src

COPY web/package.json web/package-lock.json ./
RUN npm ci

COPY web/ ./

# Path the app is served under. '/' is right while app.json keeps stripped_prefix=true,
# which makes AdaLab strip /apps/<app_url> before proxying. Build with
# --build-arg APP_BASE=/apps/<app_url>/ if that ever changes.
ARG APP_BASE=/
ENV APP_BASE=$APP_BASE
RUN npm run build

FROM nginx:alpine

COPY --from=build /src/dist /usr/share/nginx/html
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf

# Must match `port` in .adalab/local_container_1.json.
EXPOSE 8080
