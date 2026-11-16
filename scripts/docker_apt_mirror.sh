#!/usr/bin/env bash
# Select Ubuntu package repositories without changing suites or signing settings.
set -euo pipefail
mirror="${1:-official}"
apt_dir="${2:-/etc/apt}"
case "$mirror" in
    official) exit 0 ;;
    ustc) host=mirrors.ustc.edu.cn ;;
    tuna) host=mirrors.tuna.tsinghua.edu.cn ;;
    *) echo "APT_MIRROR must be official, ustc or tuna: $mirror" >&2; exit 2 ;;
esac

# HTTP supports minimal base images before ca-certificates is installed.
# APT continues to verify repository signatures using the existing keyrings.
archive='https?://(archive\.ubuntu\.com|[[:alnum:]-]+\.archive\.ubuntu\.com|security\.ubuntu\.com|mirrors\.ustc\.edu\.cn|mirrors\.tuna\.tsinghua\.edu\.cn)/ubuntu(/?)([[:space:]]|$)'
ports='https?://(ports\.ubuntu\.com|mirrors\.ustc\.edu\.cn|mirrors\.tuna\.tsinghua\.edu\.cn)/ubuntu-ports(/?)([[:space:]]|$)'
shopt -s nullglob
files=("$apt_dir/sources.list" "$apt_dir/sources.list.d/"*.list "$apt_dir/sources.list.d/"*.sources)
found=0
for file in "${files[@]}"; do
    [[ -f "$file" ]] || continue
    if grep -Eq "$archive|$ports" "$file"; then
        sed -E -i \
            -e "s#$archive#http://$host/ubuntu\\2\\3#g" \
            -e "s#$ports#http://$host/ubuntu-ports\\2\\3#g" "$file"
        found=1
    fi
done
if [[ "$found" = 0 ]]; then
    echo "No supported Ubuntu APT repositories found in $apt_dir" >&2
    exit 2
fi
echo "Ubuntu APT mirror: $mirror"
