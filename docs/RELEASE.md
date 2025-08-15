# Release workflow

## Tests

```shell
./bin/local/test_unit.sh
```

```shell
./bin/local/test_integration.sh
```

## Recreate dataset

Run all workflow before model training in `README.md`

## Bump package version

1. Bump package version in  `pyproject.toml`
2. Bump package version in  `.def` file

[//]: # (3. Bump version in `exp_config_gen.py`)

## Add tag and create release

```shell
git tag -a v0.2.0 -m "v0.2.0"
```

```shell
git push origin tag "v0.2.0" 
```

## Build

## Sync data

1. Sync `data/datasets`
2. Sync `dist/`

## Fix `.env` (server side)

## Create .sif `.env` (server side)

If there’s a bug, fix it on a hotfix branch and bump the version patch.
