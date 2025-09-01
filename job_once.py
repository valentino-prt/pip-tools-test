# job_once.py
from pathlib import Path
import argparse, os, time, random, sys

def write_prom_file(app: str, env: str, status: int, duration: float, metrics_dir: str):
    ts = int(time.time())
    body = (
        f'job_last_run_unixtime{{app="{app}",env="{env}",mode="job"}} {ts}\n'
        f'job_last_status{{app="{app}",env="{env}",mode="job"}} {status}\n'
        f'job_duration_seconds{{app="{app}",env="{env}",mode="job"}} {duration:.3f}\n'
    )
    out_dir = Path(metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{app}.prom.tmp"
    final = out_dir / f"{app}.prom"
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, final)  # atomic rename

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--app", required=True)
    p.add_argument("--env", default="mac")
    p.add_argument("--metrics-dir", default="/Users/valentinpoirot/Dev/alloy/metrics")
    p.add_argument("--simulate-sec", type=float, default=None,
                   help="Durée simulée (s). Par défaut: aléatoire 0.5–2.0")
    args = p.parse_args()

    start = time.time()
    try:
        # --- ton vrai traitement ici ---
        delay = args.simulate_sec if args.simulate_sec is not None else random.uniform(0.5, 2.0)
        time.sleep(delay)
        status = 1  # 1 = OK ; si erreur réelle: mets 0
    except Exception:
        status = 0
    finally:
        duration = time.time() - start
        write_prom_file(args.app, args.env, status, duration, args.metrics_dir)
        sys.exit(0 if status == 1 else 1)

if __name__ == "__main__":
    main()