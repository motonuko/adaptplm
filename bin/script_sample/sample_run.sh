#!/bin/bash

PROJ_ROOT=$PWD  # should be replaced

python bin/run_nested_cv_cls.py run-nested-cv-on-enz-activity-cls \
  --exp-config "${PROJ_ROOT}/data/exp_configs/exp_config_fnn.json" \
  --output-parent-dir "${PROJ_ROOT}/data/out"
