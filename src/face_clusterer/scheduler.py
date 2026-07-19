"""Day/night supervisor: capture during the day, analyze at night.

The two jobs have opposite resource profiles — capture is idle-but-continuous
and must not miss anything, analysis is a CPU burst that can take hours — so
they are separated in time rather than run concurrently. On one machine with
one camera this avoids the analysis pass starving the capture loop of CPU.

All time arithmetic lives in the pure functions at the top of this module, so
the schedule can be tested at any hour of the day without waiting for or faking
a clock. The supervisor itself is a thin loop over those decisions.

The night job is deliberately restartable: a crash or a reboot at 03:00 loses
that night's clustering, not the collected material, because capture writes
plain JPEGs and analysis is derived from them. `analysis_due` keys off the last
completed run so an interrupted night is retried, while a completed one is not
repeated.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path

from .config import AppConfig

logger = logging.getLogger(__name__)

STATE_FILE = "schedule_state.json"


# ---------------- pure scheduling logic ----------------

def parse_hhmm(value: str) -> dtime:
    """Parse 'HH:MM' into a time, rejecting anything else loudly."""
    try:
        hours, minutes = (int(part) for part in str(value).split(":"))
        return dtime(hour=hours, minute=minutes)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid time '{value}', expected HH:MM") from exc


def in_capture_window(now: datetime, start: dtime, end: dtime) -> bool:
    """Is `now` inside the capture window? Handles windows crossing midnight."""
    current = now.time()
    if start == end:
        return False                       # zero-length window: never capture
    if start < end:
        return start <= current < end
    return current >= start or current < end        # wraps past midnight


def next_transition(now: datetime, start: dtime, end: dtime) -> datetime:
    """When the current phase ends."""
    target = end if in_capture_window(now, start, end) else start
    candidate = now.replace(
        hour=target.hour, minute=target.minute, second=0, microsecond=0
    )
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def last_window_end(now: datetime, end: dtime) -> datetime:
    """The most recent moment the capture window closed (analysis start)."""
    candidate = now.replace(
        hour=end.hour, minute=end.minute, second=0, microsecond=0
    )
    if candidate > now:
        candidate -= timedelta(days=1)
    return candidate


def analysis_due(now: datetime, end: dtime, last_analysis: float | None) -> bool:
    """Should the nightly analysis run now?

    Due when the capture window has closed and no analysis has completed since
    that closing — so one run per night, retried if it failed or was cut short.
    """
    window_closed = last_window_end(now, end)
    if last_analysis is None:
        return True
    return datetime.fromtimestamp(last_analysis) < window_closed


# ---------------- supervisor ----------------

class DayNightScheduler:
    def __init__(self, cfg: AppConfig | None = None):
        self.cfg = cfg or AppConfig()
        self.start = parse_hhmm(self.cfg.schedule.capture_start)
        self.end = parse_hhmm(self.cfg.schedule.capture_end)
        self.state_path = Path(self.cfg.acquisition.dir) / STATE_FILE

    # ---------- state ----------

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Unreadable schedule state (%s) — starting fresh.", exc)
            return {}

    def _save_state(self, state: dict):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )

    # ---------- jobs ----------

    def _camera_source(self) -> int | str:
        camera = self.cfg.schedule.camera
        return int(camera) if str(camera).isdigit() else camera

    def capture_slice(self, seconds: float) -> dict:
        """Capture one session, bounded by the remaining window."""
        from .acquire import DatasetAcquisition

        acq_cfg = self.cfg.acquisition
        previous_limit = acq_cfg.max_duration_seconds
        acq_cfg.max_duration_seconds = max(1.0, seconds)
        session = datetime.now().strftime("day_%Y%m%d_%H%M%S")
        try:
            report = DatasetAcquisition(self.cfg).run(
                self._camera_source(), session=session, label="scheduled"
            )
        finally:
            acq_cfg.max_duration_seconds = previous_limit
        return report.as_summary()

    def analyze(self) -> dict:
        """Nightly job: cluster the whole dataset, then refit the classifier."""
        from .pipeline import ImagesPipeline, train_classifier

        dataset = Path(self.cfg.acquisition.dir)
        logger.info("Night job: analyzing %s", dataset)
        # A full re-analysis replaces the stored faces rather than appending,
        # otherwise every night would duplicate the entire dataset in SQLite.
        summary = ImagesPipeline(self.cfg).run(dataset, reset_store=True)

        if self.cfg.schedule.train_after_analysis:
            report = train_classifier(self.cfg)
            summary["classifier"] = {
                "trained": report.trained,
                "n_persons": report.n_persons,
                "reason": report.reason,
            }
            if report.trained and not self.cfg.classifier.enabled:
                # Training succeeded but nothing will load the model: the
                # cascade only consults the classifier when it is enabled.
                logger.warning(
                    "Classifier trained (%d persons) but classifier.enabled is "
                    "false — set it to true in config.yaml for the cascade to "
                    "use it.",
                    report.n_persons,
                )
        return summary

    # ---------- loop ----------

    def run(self, max_cycles: int = 0) -> dict:
        """Supervise indefinitely (or for `max_cycles` decisions, for testing)."""
        logger.info(
            "Scheduler: capture %s-%s from camera %s, analysis outside that window.",
            self.cfg.schedule.capture_start, self.cfg.schedule.capture_end,
            self.cfg.schedule.camera,
        )
        cycles = 0
        results = {"captures": 0, "analyses": 0, "errors": 0}

        while max_cycles == 0 or cycles < max_cycles:
            cycles += 1
            now = datetime.now()
            state = self._load_state()

            if in_capture_window(now, self.start, self.end):
                remaining = (next_transition(now, self.start, self.end) - now)
                slice_seconds = min(
                    remaining.total_seconds(),
                    self.cfg.schedule.session_minutes * 60,
                )
                try:
                    report = self.capture_slice(slice_seconds)
                    results["captures"] += 1
                    logger.info(
                        "Capture session done: %d frames saved.",
                        report["frames_saved"],
                    )
                except (IOError, OSError) as exc:
                    # A camera that is unplugged or busy must not kill the
                    # schedule: log, wait, and let the next cycle retry.
                    results["errors"] += 1
                    logger.error("Capture failed (%s) — retrying shortly.", exc)
                    self._sleep(self.cfg.schedule.retry_seconds)
                continue

            if self.cfg.schedule.analysis_enabled and analysis_due(
                now, self.end, state.get("last_analysis")
            ):
                try:
                    summary = self.analyze()
                    state["last_analysis"] = time.time()
                    state["last_analysis_summary"] = summary
                    self._save_state(state)
                    results["analyses"] += 1
                    logger.info("Night job finished: %s", summary)
                except Exception as exc:            # noqa: BLE001
                    # Analysis is the failure-prone half (models, disk, data);
                    # never let it take the capture schedule down with it.
                    results["errors"] += 1
                    logger.exception("Night job failed: %s", exc)
                    state["last_analysis_error"] = str(exc)
                    self._save_state(state)
                    self._sleep(self.cfg.schedule.retry_seconds)
                continue

            wake_at = next_transition(now, self.start, self.end)
            logger.info(
                "Idle until %s (%.1f min).",
                wake_at.strftime("%Y-%m-%d %H:%M"),
                (wake_at - now).total_seconds() / 60,
            )
            self._sleep((wake_at - now).total_seconds())

        return results

    @staticmethod
    def _sleep(seconds: float):
        time.sleep(max(0.0, seconds))


def schedule_status(cfg: AppConfig | None = None) -> dict:
    """What the scheduler would do right now, without doing it."""
    cfg = cfg or AppConfig()
    start, end = parse_hhmm(cfg.schedule.capture_start), parse_hhmm(
        cfg.schedule.capture_end
    )
    now = datetime.now()
    scheduler = DayNightScheduler(cfg)
    state = scheduler._load_state()
    capturing = in_capture_window(now, start, end)
    last = state.get("last_analysis")
    return {
        "now": now.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "capture" if capturing else "analysis",
        "window": f"{cfg.schedule.capture_start}-{cfg.schedule.capture_end}",
        "camera": cfg.schedule.camera,
        "next_transition": next_transition(now, start, end).strftime(
            "%Y-%m-%d %H:%M"
        ),
        "analysis_due": analysis_due(now, end, last),
        "last_analysis": (
            datetime.fromtimestamp(last).strftime("%Y-%m-%d %H:%M:%S")
            if last else None
        ),
        "dataset": str(Path(cfg.acquisition.dir)),
    }
