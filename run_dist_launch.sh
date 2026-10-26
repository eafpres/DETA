#!/usr/bin/env bash
set -euo pipefail
#
# This historical entry point is intentionally disabled. A one-GPU launch
# still activates DistributedDataParallel and a different training sampler,
# which made nominally comparable experiments follow different sample orders.
#
echo 'ERROR: tools/run_dist_launch.sh is archived and disabled.' >&2
echo 'For one-GPU training, invoke configs/deta_swin_ft.sh directly.' >&2
echo 'Historical launcher: archive/distributed_training/run_dist_launch.sh' >&2
exit 2
