import json
import logging
import os
import pathlib
import re
import statistics
import subprocess
from abc import ABC, abstractmethod
from collections import defaultdict

import pandas as pd
import yaml

from omnihub.process import util


class ProcessParser(ABC):
    def __init__(self, execution_dir):
        self.execution_dir = execution_dir
        self.processed_dir = f"{self.execution_dir}/processed-data"
        job_id = pathlib.Path(self.execution_dir).name
        self.log = logging.getLogger(f"Job {job_id}")

    @abstractmethod
    def parse(self):
        pass


class JobConfigParser(ProcessParser):
    def parse(self):
        with open(f"{self.execution_dir}/job.yaml", "r") as f:
            data = yaml.safe_load(f)
        records = util.flatten_dict(data["job"])
        with open(f"{self.processed_dir}/job.yaml", "w") as f:
            yaml.dump(records, f)

        tools = {tool: True for tool in data["job"].get("tools", [])}
        with open(f"{self.processed_dir}/tools.yaml", "w") as f:
            yaml.dump(tools, f)


class AppConfigParser(ProcessParser):
    def parse(self):
        with open(f"{self.execution_dir}/app.yaml", "r") as f:
            data = yaml.safe_load(f)
        records = util.flatten_dict(data)
        with open(f"{self.processed_dir}/app.yaml", "w") as f:
            yaml.dump(records, f)


class AppLogParser(ProcessParser):
    def parse(self):
        with open(f"{self.execution_dir}/job.yaml", "r") as f:
            data = yaml.safe_load(f)

        job = data["job"]
        if not "app-config" in job:
            self.log.warning("Missing app-config field in job configuration")
            return

        # Assume application log parser is available if the `parse.py`
        # executable is present in the same directory as the application
        # configuration file.
        app_dir = pathlib.Path(job["app-config"]).parent
        parser = f"{app_dir}/parse.py"
        if os.access(parser, os.X_OK):
            self.log.debug(f"Found application parser: {parser}")
            subprocess.run([parser, self.execution_dir])


class PytorchTraceVerboseParser(ProcessParser):
    def parseTraceData(self, trace_data, output_file):
        # Avoid YAML aliases
        # https://stackoverflow.com/questions/13518819/avoid-references-in-pyyaml
        yaml.Dumper.ignore_aliases = lambda *args: True

        # Extract and classify events
        main_events, kernel_events, secondary_events = self._extract_events(trace_data)

        # Convert lists to DataFrames for easier processing
        main_df = pd.DataFrame(main_events)
        kernel_df = pd.DataFrame(kernel_events)
        secondary_df = pd.DataFrame(secondary_events)

        # Match secondary events with main events to update sequence and parent IDs
        secondary_df = self._match_secondary_events(main_df, secondary_df)

        # Update kernel events using external IDs from main and secondary events
        kernel_df = self._update_kernel_events(main_df, secondary_df, kernel_df)

        # Convert DataFrames back to lists
        main_list = main_df.to_dict(orient="records")
        secondary_list = secondary_df.to_dict(orient="records")
        kernel_list = kernel_df.to_dict(orient="records")

        # Group and nest events by sequence number and direction
        grouped_events = self._group_events(main_list, secondary_list, kernel_list)

        # Save grouped events to YAML using the C-based Dumper for performance
        with open(f"{self.processed_dir}/{output_file}", "w") as f:
            yaml.dump(
                dict(grouped_events),
                f,
                default_flow_style=False,
                Dumper=yaml.CDumper,
                width=100,
            )

    def _extract_events(self, trace_data):
        main_events = []
        kernel_events = []
        secondary_events = []
        for event in trace_data.get("traceEvents", []):
            is_backward = False
            nn_stack = ""
            if event.get("cat") in ["cpu_op", "kernel"]:
                if "Call stack" in event["args"]:
                    call_stack = event["args"]["Call stack"]
                    call_stack_pairs = [
                        (k.strip(), v.strip())
                        for k, v in (
                            pair.split(":")
                            for pair in call_stack.split(";")
                            if ":" in pair
                        )
                    ]
                    # Keep only the nn.Module values to shorten the stack trace
                    nn_stack = ";".join(
                        [v for k, v in call_stack_pairs if k in "nn.Module"]
                    )
                    # Heuristic: if the call stack contains "_engine_run_backward", mark as backward.
                    for key, value in call_stack_pairs:
                        if "_engine_run_backward" in value:
                            is_backward = True
                            break

                new_event = {
                    "event_name": event.get("name"),
                    "event_id": event["args"].get("Ev Idx", -1),
                    "cat": event.get("cat"),
                    "tid": event.get("tid"),
                    "pid": event.get("pid"),
                    "time_stamp": event.get("ts"),
                    "end_time_stamp": event.get("ts") + event.get("dur", 0),
                    "Sequence number": event["args"].get("Sequence number", -1),
                    "External id": event["args"].get("External id", -1),
                    "is_backward": is_backward,
                    "Module stack": nn_stack,
                    "parent_id": -1,
                }

                if new_event["Sequence number"] != -1:
                    main_events.append(new_event)
                elif new_event["cat"] == "kernel":
                    kernel_events.append(new_event)
                else:
                    secondary_events.append(new_event)
        return main_events, kernel_events, secondary_events

    def _match_secondary_events(self, main_df, secondary_df):
        # Sort main events and group them by thread and event IDs.
        main_sorted = main_df.sort_values(["tid", "pid", "time_stamp"]).groupby(
            ["tid", "pid"]
        )
        secondary_df = secondary_df.sort_values(["tid", "pid", "time_stamp"])

        seq_updates = []
        parent_updates = []

        for (tid, pid), main_group in main_sorted:
            secondary_mask = (
                (secondary_df["tid"] == tid)
                & (secondary_df["pid"] == pid)
                & (secondary_df["Sequence number"] == -1)
            )
            secondary_subset = secondary_df[secondary_mask]
            if secondary_subset.empty or main_group.empty:
                continue

            kvals = main_group[
                ["time_stamp", "end_time_stamp", "Sequence number", "event_id"]
            ].values

            kidx = 0
            for uidx, urow in secondary_subset.iterrows():
                while kidx < len(kvals) and kvals[kidx][1] < urow["time_stamp"]:
                    kidx += 1
                if (
                    kidx < len(kvals)
                    and kvals[kidx][0] <= urow["time_stamp"]
                    and kvals[kidx][1] >= urow["end_time_stamp"]
                ):
                    seq_updates.append((uidx, kvals[kidx][2]))
                    parent_updates.append((uidx, kvals[kidx][3]))

        for idx, seq_val in seq_updates:
            secondary_df.at[idx, "Sequence number"] = seq_val
        for idx, parent_val in parent_updates:
            secondary_df.at[idx, "parent_id"] = parent_val
        return secondary_df

    def _update_kernel_events(self, main_df, secondary_df, kernel_df):
        # Combine main and secondary events to map External id to sequence number.
        ext_df = pd.concat(
            [
                main_df[["External id", "Sequence number"]],
                secondary_df[["External id", "Sequence number"]],
            ]
        ).drop_duplicates()

        kernel_df = kernel_df.merge(
            ext_df,
            on="External id",
            how="left",
            suffixes=("", "_updated"),
        )
        kernel_df["Sequence number"] = kernel_df["Sequence number_updated"]
        kernel_df = kernel_df.drop(columns=["Sequence number_updated"])

        # Map External id to update parent_id from main events.
        main_map = (
            main_df.sort_values("time_stamp")
            .drop_duplicates(subset=["External id"], keep="last")
            .rename(columns={"event_id": "known_parent_id"})
        )
        main_map = main_map[["External id", "known_parent_id"]]
        kernel_df = kernel_df.merge(
            main_map, on="External id", how="left", suffixes=("", "_updated")
        )
        kernel_df["parent_id"] = kernel_df["known_parent_id"].fillna(-1).astype(int)
        kernel_df = kernel_df.drop(columns=["known_parent_id"])

        # Map External id to update parent_id from secondary events.
        secondary_map = (
            secondary_df.sort_values("time_stamp")
            .drop_duplicates(subset=["External id"], keep="last")
            .rename(columns={"event_id": "known_parent_id"})
        )
        secondary_map = secondary_map[["External id", "known_parent_id"]]
        kernel_df = kernel_df.merge(
            secondary_map, on="External id", how="left", suffixes=("", "_updated")
        )
        kernel_df["parent_id"] = kernel_df["known_parent_id"].fillna(-1).astype(int)
        kernel_df = kernel_df.drop(columns=["known_parent_id"])

        return kernel_df

    def _group_events(self, main_list, secondary_list, kernel_list):
        # Combine main, secondary, and kernel events.
        all_events = main_list + secondary_list + kernel_list

        # Build a mapping from sequence number to module using main events only.
        seq_to_module = {}
        for event in main_list:
            seq = (
                int(float(event["Sequence number"]))
                if event["Sequence number"] is not None
                else -1
            )
            if event["Module stack"] and not event["is_backward"]:
                seq_to_module[seq] = event["Module stack"]

        grouped = defaultdict(lambda: {"Forward": [], "Backward": []})
        for event in all_events:
            seq = (
                int(float(event["Sequence number"]))
                if pd.notna(event["Sequence number"])
                else -1
            )
            torch_module = seq_to_module.get(seq, "Unknown")
            key = f"Seq: {seq}; Stack: {torch_module}"
            key = key if len(key) <= 120 else key[:120]
            direction = "Backward" if event["is_backward"] else "Forward"
            grouped[key][direction].append(event)

        # For each subgroup, sort events by time_stamp and build a nested structure.
        for key, directions in grouped.items():
            for direction, events in directions.items():
                sorted_events = sorted(events, key=lambda x: x["time_stamp"])
                grouped[key][direction] = self._build_nested_structure(sorted_events)

        return grouped

    def _build_nested_structure(self, events):
        # Create a lookup for events by event_id.
        event_dict = {event["event_id"]: event for event in events}
        root_events = []
        for event in events:
            parent = event["parent_id"]
            if parent == -1:
                root_events.append(event)
            else:
                parent_event = event_dict.get(parent)
                if parent_event:
                    parent_event.setdefault("successors", []).append(event)

        # Recursively format each event.
        def format_event(event):
            formatted = {event["cat"]: event["event_name"]}
            if event.get("successors"):
                formatted["successors"] = [
                    format_event(child) for child in event["successors"]
                ]
            return formatted

        return [format_event(event) for event in root_events]

    def parse(self):
        pytorch_trace_dir = f"{self.execution_dir}/tools/pytorch-trace-verbose"
        trace_files = pathlib.Path(pytorch_trace_dir).glob("*.pt.trace.json")
        for trace_file in trace_files:
            with open(trace_file, "r") as f:
                trace_data = json.load(f)
            if not trace_data:
                self.log.warning(
                    f"Unable to parse PyTorch trace data from {trace_file}"
                )
                continue
            output_file = f"{trace_file.stem}.callgraph.yaml"
            self.parseTraceData(trace_data, output_file)


class OmnihubMonitorParser(ProcessParser):
    def parse(self):
        omnihub_monitor_dir = f"{self.execution_dir}/tools/omnihub-monitor"
        data = []
        for p in pathlib.Path(omnihub_monitor_dir).glob("*.json"):
            with open(p, "r") as f:
                rank_data = json.load(f)
            data.append(rank_data)

        if len(data) == 0:
            self.log.warning("Unable to parse default OmniHub monitor data")
            return

        start = min([i["StartTime"] for i in data])
        end = max([i["EndTime"] for i in data])
        durations = [i["Duration"] for i in data]
        energy = sum([i["TotalEnergy"] for i in data])

        records = {
            "Time (s)": (end - start),
            "Rank mean time (s)": statistics.mean(durations),
            "Rank min time (s)": min(durations),
            "Rank max time (s)": max(durations),
            "GPU energy (kWh)": energy,
        }

        with open(f"{self.processed_dir}/omnihub-monitor.yaml", "w") as f:
            yaml.dump(records, f)


# Current implementation to load Omnistat data relies on the text report,
# which isn't the most machine-readable format, and so it is more complex than
# it should be. This is only a temporary solution and we should work on making
# all the basic Omnistat data relevant for Omnihub easily parseable.
class OmnistatReportParser(ProcessParser):
    def __init__(self, execution_dir, name="omnistat"):
        super().__init__(execution_dir)
        self.variant_name = name

    def parse(self):
        report_file = f"{self.execution_dir}/tools/{self.variant_name}/report.txt"
        with open(report_file, "r") as f:
            report = f.readlines()

        gpu_pattern = re.compile(r"[\s]+[0-9]+[\s]+\|")
        gpu_table = [line for line in report if re.match(gpu_pattern, line)]
        gpu_data = []
        for row in gpu_table:
            row = row.strip().split("|")
            utilization = row[1].split()
            memory = row[2].split()
            gpu = {
                "util-max": float(utilization[0]),
                "util-mean": float(utilization[1]),
                "mem-max": float(memory[0]),
                "mem-mean": float(memory[1]),
            }
            gpu_data.append(gpu)

        if len(gpu_data) == 0:
            self.log.warning("Unable to parse GPU data from Omnistat report")
            return

        records = {
            "GPU max utilization (%)": max([i["util-max"] for i in gpu_data]),
            "GPU mean utilization (%)": statistics.mean(
                [i["util-mean"] for i in gpu_data]
            ),
            "GPU max memory (%)": max([i["mem-max"] for i in gpu_data]),
            "GPU mean memory (%)": statistics.mean([i["mem-mean"] for i in gpu_data]),
        }

        energy_pattern = re.compile(
            r"Approximate Total GPU Energy Consumed = ([0-9]*\.[0-9]+|[0-9]+) kWh"
        )
        for line in report:
            match = re.match(energy_pattern, line)
            if match:
                records["GPU energy (kWh)"] = match.group(1)

        with open(f"{self.processed_dir}/{self.variant_name}.yaml", "w") as f:
            yaml.dump(records, f)


class JobReportParser(ProcessParser):
    def parse(self):
        with open(f"{self.execution_dir}/job-report.yaml", "r") as f:
            data = yaml.safe_load(f)
        duration_ms = data["execute_end_ms"] - data["execute_start_ms"]
        data["execute_duration_s"] = duration_ms / 1000
        with open(f"{self.processed_dir}/job-report.yaml", "w") as f:
            yaml.dump(data, f)


class RocprofStatsParser(ProcessParser):
    def parse(self):
        self.parse_columns = ["Percentage", "TotalDurationNs"]

        # Store all per-rank dataframes to measure aggregated stats.
        all_dfs = defaultdict(list)

        # Generate per-rank data files with top kernel stats.
        rank_dirs = pathlib.Path(
            f"{self.execution_dir}/tools/rocprofv3-stats/"
        ).iterdir()
        for rank_dir in rank_dirs:
            rank = rank_dir.name
            columns = self._parse_rank(rank_dir)
            for column, column_df in columns.items():
                data = column_df.to_dict()[rank]
                path = f"{self.processed_dir}/rocprofv3-stats.{column}.rank-{rank}.json"
                with open(path, "w") as f:
                    json.dump(data, f)
                all_dfs[column].append(column_df)

        # Generate aggregated mean and stddev files.
        for column, dfs in all_dfs.items():
            df = pd.concat(dfs, axis=1)

            # Convert NaNs to zeros to calculate aggregated stats, assume that
            # kernels that aren't present in certain ranks are negligible.
            # While not entirely accurate, it's an initial approximation.
            df = df.fillna(0)

            mean = df.mean(axis=1).to_frame(name="mean")
            data = mean.to_dict()["mean"]
            path = f"{self.processed_dir}/rocprofv3-stats.{column}.mean.json"
            with open(path, "w") as f:
                json.dump(data, f)

            std = df.std(axis=1).to_frame(name="std")
            data = std.to_dict()["std"]
            path = f"{self.processed_dir}/rocprofv3-stats.{column}.std.json"
            with open(path, "w") as f:
                json.dump(data, f)

    def _parse_rank(self, rank_dir):
        rank = rank_dir.name
        stats_files = list(rank_dir.glob("*_kernel_stats.csv"))

        if len(stats_files) == 0:
            self.log.warning(f"Unable to find kernel stats for rank {rank}")
            return

        # Assume we only have one trace file per rank.
        stats_file = stats_files[0]

        # Select top 10 kernels.
        df = pd.read_csv(stats_file)
        df = df.head(10)

        columns = {}
        for column in self.parse_columns:
            column_df = df[["Name", column]]
            column_df = column_df.set_index(column_df.columns[0])
            # Rename column to rank so it's easier to aggregate results;
            # we alredy keep track of the column name in the dictionary.
            column_df.columns = [rank]
            columns[column] = column_df

        return columns
