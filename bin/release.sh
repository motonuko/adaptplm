#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Checkout develop branch and rebase pull
git checkout develop
git pull --rebase

# Check for merge conflicts
if [ -n "$(git ls-files -u)" ]; then
  echo "Merge conflicts detected. Please resolve them before proceeding."
  exit 1
fi

# Extract version from pyproject.toml
VERSION=$(grep '^version =' pyproject.toml | sed -E 's/version = "(.*)"/\1/')

if [ -z "$VERSION" ]; then
  echo "Failed to extract version from pyproject.toml"
  exit 1
fi

# Tagging the release
TAG="v$VERSION"
git tag -a "$TAG" -m "$TAG"
git push origin "$TAG"

echo "Release $TAG created and pushed successfully."
