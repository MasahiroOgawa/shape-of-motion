#!/bin/bash
# Monitor GPU and system memory during training
# AGGRESSIVE monitoring - kills EARLY before system freezes

TRAINING_PID=$1
GPU_THRESHOLD=75       # Kill at 75% GPU (~9.2GB of 12GB) - leave headroom
SYS_MEM_THRESHOLD=70   # Kill at 70% system RAM - leave headroom
GPU_TOTAL_MB=12282     # RTX 4080 12GB
CHECK_INTERVAL=1       # Check every 1 second - fast response

if [ -z "$TRAINING_PID" ]; then
    echo "Usage: $0 <training_pid>"
    exit 1
fi

echo "=== AGGRESSIVE MEMORY MONITOR ==="
echo "Monitoring PID $TRAINING_PID"
echo "GPU kill threshold: ${GPU_THRESHOLD}% (~$((GPU_TOTAL_MB * GPU_THRESHOLD / 100))MB)"
echo "SysMem kill threshold: ${SYS_MEM_THRESHOLD}%"
echo "Check interval: ${CHECK_INTERVAL}s"
echo "=================================="

# Track consecutive high readings to avoid false positives on brief spikes
GPU_HIGH_COUNT=0
SYS_HIGH_COUNT=0
KILL_AFTER_CONSECUTIVE=2  # Kill after 2 consecutive high readings (2 seconds)

while kill -0 "$TRAINING_PID" 2>/dev/null; do
    # GPU memory
    GPU_USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    GPU_PCT=$((GPU_USED_MB * 100 / GPU_TOTAL_MB))

    # System memory
    SYS_MEM_PCT=$(free | awk '/Mem:/ {printf "%.0f", $3/$2 * 100}')

    TIMESTAMP=$(date '+%H:%M:%S')

    # Warning levels
    if [ "$GPU_PCT" -ge 60 ] || [ "$SYS_MEM_PCT" -ge 55 ]; then
        echo "[$TIMESTAMP] GPU: ${GPU_USED_MB}MB/${GPU_TOTAL_MB}MB (${GPU_PCT}%) | SysMem: ${SYS_MEM_PCT}%"
    fi

    # GPU check
    if [ "$GPU_PCT" -ge "$GPU_THRESHOLD" ]; then
        GPU_HIGH_COUNT=$((GPU_HIGH_COUNT + 1))
        echo "[$TIMESTAMP] WARNING: GPU ${GPU_PCT}% >= ${GPU_THRESHOLD}% (count: $GPU_HIGH_COUNT/$KILL_AFTER_CONSECUTIVE)"
        if [ "$GPU_HIGH_COUNT" -ge "$KILL_AFTER_CONSECUTIVE" ]; then
            echo "!!! KILLING training - GPU memory too high: ${GPU_USED_MB}MB (${GPU_PCT}%)"
            kill -9 "$TRAINING_PID" 2>/dev/null
            # Also kill any child processes
            pkill -9 -P "$TRAINING_PID" 2>/dev/null
            sleep 1
            echo "KILLED_GPU"
            exit 1
        fi
    else
        GPU_HIGH_COUNT=0
    fi

    # System memory check
    if [ "$SYS_MEM_PCT" -ge "$SYS_MEM_THRESHOLD" ]; then
        SYS_HIGH_COUNT=$((SYS_HIGH_COUNT + 1))
        echo "[$TIMESTAMP] WARNING: SysMem ${SYS_MEM_PCT}% >= ${SYS_MEM_THRESHOLD}% (count: $SYS_HIGH_COUNT/$KILL_AFTER_CONSECUTIVE)"
        if [ "$SYS_HIGH_COUNT" -ge "$KILL_AFTER_CONSECUTIVE" ]; then
            echo "!!! KILLING training - System memory too high: ${SYS_MEM_PCT}%"
            kill -9 "$TRAINING_PID" 2>/dev/null
            pkill -9 -P "$TRAINING_PID" 2>/dev/null
            sleep 1
            echo "KILLED_SYSMEM"
            exit 2
        fi
    else
        SYS_HIGH_COUNT=0
    fi

    sleep "$CHECK_INTERVAL"
done

echo "Training process $TRAINING_PID finished normally."
echo "FINISHED_OK"
