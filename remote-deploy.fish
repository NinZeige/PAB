#!/usr/bin/env fish

set origin_path (status dirname)

set -g ignore_expr '.venv/' \
    '__pycache__/' \
    '.git/' \
    'pyproject.toml' \
    'uv.lock' \
    'checkpoints/'

set -g rsync_ignore_expr

for expr in $ignore_expr
    set -a rsync_ignore_expr (printf -f'- %s' $expr)
end

if test $origin_path != (status dirname)
    echo "Panic" >&2
end

ruff format
rsync -r -P -e 'ssh -p 28367' $rsync_ignore_expr (status dirname)  ninzeige@region-42.seetacloud.com:~/mclip-pab/

