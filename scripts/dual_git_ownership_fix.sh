#!/bin/sh

# WARNING: only used as a temp fix while using root user in docker to fix git dual ownership issue that mess up wrong output dir. This fix script should oly be used as an entrypoint when running the container, not in host environment.
git config --global --add safe.directory /home
