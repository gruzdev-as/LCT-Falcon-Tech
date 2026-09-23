# Build context is frontend/, not the repo root like backend/: the SPA imports nothing
# from common/, and a root context would ship .venv/ and data/ to the daemon.

FROM node:26.8.1-alpine3.23 AS build

WORKDIR /app

COPY package.json package-lock.json ./
# Not --omit=dev: vite, typescript and tailwind are devDependencies.
RUN npm ci

COPY . .

# Vite inlines VITE_* at build time, so this must be a build arg — changing it needs a
# rebuild, not a restart.
ARG VITE_DEMO_MODE=false
ENV VITE_DEMO_MODE=$VITE_DEMO_MODE
RUN npm run build

FROM nginx:1.29.3-alpine AS runtime

COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

# No USER: the nginx image runs a root master with unprivileged workers, which is what
# lets it bind :80.
EXPOSE 80
