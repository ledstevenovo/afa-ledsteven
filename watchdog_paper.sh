#!/bin/bash
# Watchdog: keep the paper-setting AFA run alive until the morning stop time.
# Runs as a loop (no cron dependency) inside tmux session afa_wd.
# Restarts the wrapper (which itself resumes from the last step checkpoint)
# if both the wrapper and the python process are gone before ~09:05.
DEADLINE=$(date -d "today 09:05" +%s); [ "$(date +%H%M)" -ge "0905" ] && DEADLINE=$(date -d "tomorrow 09:05" +%s)
cd /root/bayes-tmp/afa || exit 1
LOG=/root/bayes-tmp/afa-assets/logs/watchdog.log
mkdir -p "$(dirname "$LOG")"
echo "[$(date '+%F %T')] watchdog started (pid $$), deadline $(date -d @$DEADLINE '+%F %T')" >> "$LOG"

while true; do
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        echo "[$(date '+%F %T')] deadline reached, watchdog exiting" >> "$LOG"
        exit 0
    fi
    if pgrep -f "[t]rain_paper_setting.py" > /dev/null; then
        : # training alive
    elif pgrep -f "[r]un_paper_setting.sh" > /dev/null; then
        : # wrapper alive, waiting for retry
    else
        echo "[$(date '+%F %T')] RUN NOT FOUND -> restarting wrapper" >> "$LOG"
        if tmux has-session -t afa_run 2>/dev/null; then
            tmux send-keys -t afa_run 'cd /root/bayes-tmp/afa && bash run_paper_setting.sh' Enter
        else
            tmux new-session -d -s afa_run -c /root/bayes-tmp/afa "bash run_paper_setting.sh"
        fi
        sleep 60
    fi
    sleep 240
done
