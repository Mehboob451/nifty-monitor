#!/usr/bin/env bash
# One-shot installer: replaces the old files in this repo with Nifty Monitor and pushes to GitHub.
set -e
rm -f nifty-monitor.zip setup.sh
git rm -q --ignore-unmatch collector.py dashboard.py main.py index.html run.yml data.json history.json >/dev/null 2>&1 || true
rm -f collector.py dashboard.py main.py index.html run.yml
git pull --rebase --autostash -q 2>/dev/null || true
git add -A
git commit -q -m "Nifty Monitor: forecasts, options data, alerts and scoreboard" || echo "(nothing new to commit)"
git push origin HEAD
echo ""
echo "DONE. Files are on GitHub. Now do steps 2 to 5 of the guide (secret, permissions, first run, Pages)."
