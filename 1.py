#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sync_exchange_time_daemon.py
24/7 синхронизация системного времени с серверным временем биржи (по умолчанию Binance).
- Опрашивает /api/v3/time, учитывает половину RTT.
- Корректирует системное время в UTC: Windows (SetSystemTime), Linux (date -u -s / timedatectl).
- Порог применения, интервал опроса, экспоненциальный бэкофф, журналирование.
"""

import argparse
import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    print("Установите зависимость: pip install requests", file=sys.stderr)
    sys.exit(1)

BINANCE_TIME_URL = "https://api.binance.com/api/v3/time"  # {"serverTime": <ms since epoch>}

def is_admin():
    if os.name == "nt":
        try:
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            return False
    else:
        return os.geteuid() == 0

def fetch_server_time_ms(session: requests.Session, url: str) -> tuple[int, float]:
    t0 = time.monotonic_ns()
    resp = session.get(url, timeout=5)
    t1 = time.monotonic_ns()
    resp.raise_for_status()
    data = resp.json()
    server_ms = int(data["serverTime"])
    rtt_ms = (t1 - t0) / 1_000_000.0
    corrected_ms = server_ms + rtt_ms / 2.0
    return int(corrected_ms), rtt_ms

def windows_set_utc(dt_utc: datetime):
    class SYSTEMTIME(ctypes.Structure):
        _fields_ = [
            ("wYear", ctypes.c_ushort),
            ("wMonth", ctypes.c_ushort),
            ("wDayOfWeek", ctypes.c_ushort),
            ("wDay", ctypes.c_ushort),
            ("wHour", ctypes.c_ushort),
            ("wMinute", ctypes.c_ushort),
            ("wSecond", ctypes.c_ushort),
            ("wMilliseconds", ctypes.c_ushort),
        ]
    st = SYSTEMTIME()
    st.wYear = dt_utc.year
    st.wMonth = dt_utc.month
    st.wDay = dt_utc.day
    st.wHour = dt_utc.hour
    st.wMinute = dt_utc.minute
    st.wSecond = dt_utc.second
    st.wMilliseconds = int(dt_utc.microsecond / 1000)
    st.wDayOfWeek = 0
    if ctypes.windll.kernel32.SetSystemTime(ctypes.byref(st)) == 0:
        err = ctypes.GetLastError()
        raise OSError(f"SetSystemTime failed, GetLastError={err}")

def linux_set_utc(epoch_seconds: int, may_toggle_ntp: bool, restore_ntp: bool):
    # Try GNU date first (works even when NTP enabled on большинстве дистрибутивов)
    date_bin = shutil.which("date")
    if date_bin:
        r = subprocess.run([date_bin, "-u", "-s", f"@{epoch_seconds}"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode == 0:
            return
    # Fallback: timedatectl (может требовать отключения NTP)
    timedatectl = shutil.which("timedatectl")
    if not timedatectl:
        raise FileNotFoundError("Neither 'date' nor 'timedatectl' found to set system time")
    ntp_was_on = False
    if may_toggle_ntp:
        # timedatectl status --no-pager --property=NTP
        status = subprocess.run([timedatectl, "show"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if status.returncode == 0 and "NTPSynchronized=" in status.stdout:
            # Определить, активна ли синхронизация (грубая эвристика)
            ntp_was_on = "NTP=yes" in status.stdout or "NTPSynchronized=yes" in status.stdout
        subprocess.run([timedatectl, "set-ntp", "false"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    dt_utc = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    ts = dt_utc.strftime("%Y-%m-%d %H:%M:%S")
    r2 = subprocess.run([timedatectl, "set-time", ts], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r2.returncode != 0:
        raise OSError(f"timedatectl set-time failed: {r2.stderr.strip()}")
    if may_toggle_ntp and restore_ntp and ntp_was_on:
        subprocess.run([timedatectl, "set-ntp", "true"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def apply_system_time(target_ms: int, os_name: str, may_toggle_ntp: bool, restore_ntp: bool):
    epoch_sec = target_ms // 1000
    if "windows" in os_name:
        dt_utc = datetime.fromtimestamp(epoch_sec, tz=timezone.utc)
        windows_set_utc(dt_utc)
    elif "linux" in os_name:
        linux_set_utc(epoch_sec, may_toggle_ntp, restore_ntp)
    else:
        raise NotImplementedError(f"Unsupported OS: {os_name}")

def main():
    ap = argparse.ArgumentParser(description="24/7 sync system time with exchange server time (Binance)")
    ap.add_argument("--url", type=str, default=BINANCE_TIME_URL, help="Endpoint returning JSON with serverTime in ms")
    ap.add_argument("--interval", type=int, default=300, help="Polling interval seconds")
    ap.add_argument("--threshold-ms", type=int, default=150, help="Apply only if |drift| exceeds this threshold")
    ap.add_argument("--max-backoff", type=int, default=900, help="Max backoff seconds on repeated errors")
    ap.add_argument("--no-apply", action="store_true", help="Monitor drift only (no system time changes)")
    ap.add_argument("--toggle-ntp", action="store_true", help="Linux: allow temporarily disabling NTP to set time")
    ap.add_argument("--restore-ntp", action="store_true", help="Linux: re-enable NTP if it was on")
    args = ap.parse_args()

    os_name = platform.system().lower()
    if not is_admin() and not args.no_apply:
        print(json.dumps({"level": "error", "msg": "Administrator/root privileges required to set system time"}))
        sys.exit(2)

    backoff = args.interval
    with requests.Session() as s:
        while True:
            loop_started = time.time()
            try:
                server_ms, rtt_ms = fetch_server_time_ms(s, args.url)
                local_ms = int(time.time() * 1000)
                drift_ms = server_ms - local_ms
                payload = {
                    "ts": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "server_utc": datetime.utcfromtimestamp(server_ms/1000).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "local_utc": datetime.utcfromtimestamp(local_ms/1000).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "rtt_ms": round(rtt_ms, 1),
                    "drift_ms": drift_ms
                }
                if args.no_apply or abs(drift_ms) < args.threshold_ms:
                    payload["action"] = "noop"
                    print(json.dumps(payload, ensure_ascii=False))
                    backoff = args.interval
                else:
                    apply_system_time(server_ms, os_name, args.toggle_ntp, args.restore_ntp)
                    new_local_ms = int(time.time() * 1000)
                    payload["action"] = "set"
                    payload["new_local_utc"] = datetime.utcfromtimestamp(new_local_ms/1000).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    payload["residual_ms"] = server_ms - new_local_ms
                    print(json.dumps(payload, ensure_ascii=False))
                    backoff = args.interval
            except Exception as e:
                print(json.dumps({"level": "error", "msg": str(e)}))
                # экспоненциальный бэкофф до max-backoff
                backoff = min(max(1, backoff * 2), max(1, args.max_backoff))
            # спать до следующего цикла, учитывая время выполнения
            elapsed = time.time() - loop_started
            sleep_for = max(1.0, backoff - elapsed)
            time.sleep(sleep_for)

if __name__ == "__main__":
    main()
