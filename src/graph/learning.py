import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class LogFile:
    time: datetime
    path: Path
    parsed: dict
    lines: list[dict] = None
    hours: int = 12

    def __post_init__(self):
        self.lines = self.read_lines()

    @property
    def label(self):
        return "".join([f"[{key}-{value}]" for key, value in self.parsed.items()])

    def read_lines(self):
        text = self.path.read_text("utf-8")
        result_list = []
        start_time = None
        for line in text.split("\n"):
            if grp := re.search(r"hours : (\d+)", line):
                self.hours = max(int(grp.group(1)), 1)
            if "GAMEOVER" not in line:
                continue
            parsed = parse_line(line)
            if parsed:
                if start_time is None:
                    start_time = parsed["time"]
                rel_time = (parsed["time"] - start_time).total_seconds() // interval
                result_list.append(
                    {
                        "rel_time": rel_time,
                        "time": parsed["time"],
                        "score": parsed["score"],
                        "init_eval": parsed["init_eval"],
                        "init_eval_2": parsed.get("init_eval_2", 0),
                    }
                )
        if not result_list:
            raise ValueError(f"Invalid log file: {self.path.name}")
        print(
            f"Loaded {len(result_list)} lines from {self.path.name}\nTotal seconds: {(result_list[-1]['time'] - start_time).total_seconds()}"
        )
        return result_list


plt.rcParams["font.size"] = 14
BASE_DIR = Path(__file__).parent.parent.parent
LOG_DIR = BASE_DIR / "log"
interval = 300


def parse_path(path: Path) -> LogFile:
    if match := re.match(
        r"(?:\[(.*?)\]_)*(\d{8}T\d{6})_\[(.*)\]$", path.stem, re.IGNORECASE
    ):
        filename, dtm, kv_pairs = match.groups()
        time = datetime.strptime(dtm, "%Y%m%dT%H%M%S")
        kv_dict = dict(pair.split("-") for pair in kv_pairs.split("]["))
        return LogFile(
            time=time,
            path=path,
            parsed=kv_dict,
        )
    raise ValueError(f"Invalid log file name: {path.stem}")


def parse_line(line: str):
    match = re.match(
        r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}):INFO:GAMEOVER:\s+thread_id=(\d+)\s+count=(\d+)\s+bd\.score=(\d+)\s+turn=(\d+).*queue_size=(\d+)\s+init_eval_1=([\d.]+)\s+init_eval_2=([\d.]+)",
        line,
    )
    if match:
        return {
            "time": datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%S"),
            "thread_id": int(match.group(2)),
            "count": int(match.group(3)),
            "score": int(match.group(4)),
            "turn": int(match.group(5)),
            "queue_size": int(match.group(6)),
            "init_eval": float(match.group(7)),
            "init_eval_2": float(match.group(8)),
        }
    match2 = re.match(
        r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}):INFO:GAMEOVER:\s+thread_id=(\d+)\s+count=(\d+)\s+bd\.score=(\d+)\s+turn=(\d+).*queue_size=(\d+)\s+init_eval_1=([\d.]+)",
        line,
    )
    if match2:
        return {
            "time": datetime.strptime(match2.group(1), "%Y-%m-%dT%H:%M:%S"),
            "thread_id": int(match2.group(2)),
            "count": int(match2.group(3)),
            "score": int(match2.group(4)),
            "turn": int(match2.group(5)),
            "queue_size": int(match2.group(6)),
            "init_eval": float(match2.group(7)),
        }
    return None


def aggregate_by_interval(data, interval):
    grouped = defaultdict(list)

    # rel_timeをintervalで区切ってグループ化
    for entry in data:
        grouped[entry["rel_time"]].append(entry)

    # 平均を計算
    result = []
    for bucket in sorted(grouped.keys()):
        entries = grouped[bucket]
        scores = [e["score"] for e in entries]
        init_evals = [e["init_eval"] for e in entries]
        init_eval_2 = [e["init_eval_2"] for e in entries]
        result.append(
            {
                "interval": int(bucket * interval),
                "score": np.mean(scores),
                "init_eval": np.mean(init_evals),
                "init_eval_2": np.mean(init_eval_2),
            }
        )

    return result


plt.figure(
    # figsize=(12, 8)
)
log_files = [parse_path(log) for log in LOG_DIR.glob("*.log")]
structured_logs = defaultdict(list)
for log_file in log_files:
    structured_logs[log_file.label].append(log_file)

x_limit = 0
for i, (label, logs) in enumerate(structured_logs.items()):
    if len(logs) == 0:
        raise ValueError(f"Invalid log file: {label}")
    logs.sort(key=lambda x: x.time)
    if len(logs) >= 2:
        log_agg = []
        tmp = 0
        for j, log in enumerate(logs):
            adjust = (
                timedelta(hours=logs[max(0, j - 1)].hours).total_seconds() // interval
            )
            tmp += log.hours
            log_agg.extend(
                aggregate_by_interval(
                    [
                        {
                            "rel_time": line["rel_time"] + adjust * j,
                            "score": line["score"],
                            "init_eval": line["init_eval"],
                            "init_eval_2": line["init_eval_2"],
                        }
                        for line in log.lines
                    ],
                    interval,
                )
            )
        print(log_agg[-1])
    else:
        log = logs[0]
        log_agg = aggregate_by_interval(log.lines, interval)
        tmp = log.hours
    x_limit = max(x_limit, tmp)
    x = [entry["interval"] / 3600 for entry in log_agg]
    y_score = [entry["score"] for entry in log_agg]
    y_init_eval = [entry["init_eval"] for entry in log_agg]
    y_init_eval_2 = [entry["init_eval_2"] for entry in log_agg]
    plt.plot(
        x,
        y_score,
        label=label,
        color="C" + str(i),
        linestyle="-",
    )
    plt.plot(
        x,
        y_init_eval,
        label=label + "_init_eval",
        color="C" + str(i),
        linestyle="--",
    )
    # plt.plot(
    #     x,
    #     y_init_eval_2,
    #     label=label + "_init_eval_2",
    #     color="C" + str(i),
    #     linestyle=":",
    # )
    plt.xlabel("Time (hours)")
    plt.ylabel("Score")
    plt.title("Average Score over Time")
    plt.legend()
plt.ylim(0, 6500)
plt.xlim(0, x_limit)
plt.grid()
save_dir = Path() / "dist"
save_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(save_dir / "score.pdf", bbox_inches="tight")
plt.show()
