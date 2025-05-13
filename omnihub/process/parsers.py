import json
import logging
import os
import pathlib
import re
import statistics
import subprocess
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from operator import itemgetter

import orjson
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


class PytorchTraceParser(ProcessParser):
    def parseTraceData(self, trace_data, output_file):
        # Heuristics: for each (pid, tid) group of events, do the below
        # 1. Ensure that each event has the required fields -- tid, pid, name, and args
        # 2. Separate out all the kernel events because they will be merged later.
        # 3. Processing of 'user_annotation' events:
        # For each event, if it is a user annotation, push the name onto the annotation stack if it starts with 's:',
        # or pop the name off the annotation stack if it starts with 'f:'. The nn_stack field of each forward event is the last element of the annotation stack.
        # 4. Classification of 'backward' events:
        # If the name contains the backward root prefix and not backward accumulate grad,
        # then begin counting backward events based on the time stamp and duration of the first backward event.
        # 4a. If one backward mode is followed by another, extend the existing backward time window instead of closing and reopening it.
        # 4b. Each backward event in a time window is marked as backward and has the same sequence number as the first backward event of that window.
        # 4c. If a backward event is not the root event of a sequence group, it is a child of the root event of that sequence group.
        # 5. Classification of 'forward' events:
        # The rest of the events are obviously all forward events.
        # A forward event in the (pid, tid) group beginning with a new sequence number is the root event of this sequence group. Begin counting forward events
        # based on the time stamp and duration of the root forward event.
        # 5a. If one forward mode is followed by another, extend the existing forward time window instead of closing and reopening it.
        # 5b. Each forward event in a time window is marked as forward and has the same sequence number as the root forward event of that window.
        # 5c. If a forward event is not the root event of a sequence group, it is a child of the root event of that sequence group.
        # 6. Merge the kernel events with the forward and backward events based on the external id.
        # 6a. The kernel event then inherits the sequence number of the forward or backward event and the forward or backward event is marked as the parent of the kernel event.
        # 7. Group the events by sequence number and direction (forward or backward) and build a nested structure.
        # 8. Write the output to a YAML or JSON file.

        # Collect kernel events and filter others
        kernel_events = self._extract_kernel_events(trace_data)
        trace_data["traceEvents"] = [
            e
            for e in trace_data.get("traceEvents", [])
            if e.get("cat") in ["user_annotation", "cpu_op"]
            and all(k in e for k in ("pid", "tid", "name", "args"))
        ]

        grouped_by_seq_mode = self._process_events(
            trace_data["traceEvents"], kernel_events
        )
        self._write_grouped_output(grouped_by_seq_mode, output_file)

    def _extract_kernel_events(self, trace_data):
        return [
            {
                "event_name": e["name"],
                "event_id": e["args"].get("Ev Idx", -1),
                "cat": e["cat"],
                "tid": e["tid"],
                "pid": e["pid"],
                "time_stamp": e["ts"],
                "end_time_stamp": e["ts"] + e.get("dur", 0),
                "Sequence number": e["args"].get("Sequence number", -1),
                "External id": e["args"].get("External id", -1),
                "mode": "Forward",
                "nn_stack": "",
                "parent_id": -1,
            }
            for e in trace_data.get("traceEvents", [])
            if e.get("cat") == "kernel"
        ]

    def _process_events(self, events, kernel_events):
        BACKWARD_ROOT_PREFIX = "autograd::engine::evaluate_function:"

        events.sort(key=itemgetter("pid", "tid", "ts"))

        annotation_stack = []
        external_map = {}
        seq_module_map = defaultdict(lambda: "Unknown")

        class Mode(Enum):
            FORWARD = "Forward"
            BACKWARD = "Backward"

            def __str__(self):
                return self.value

        active_mode, seq_end, current_seq, current_root = Mode.FORWARD, -1, -1, -1
        grouped_by_seq_mode = defaultdict(lambda: {"Forward": [], "Backward": []})

        for e in events:
            if e["cat"] == "user_annotation":
                current_seq = self._update_annotation_stack(
                    e["name"], annotation_stack, current_seq
                )
                continue

            new_event = self._build_cpu_op_event(
                e, annotation_stack, str(Mode.FORWARD), current_seq
            )
            name, start, duration = e["name"], e["ts"], e.get("dur", 0)

            # update sequence number, parent id, and mode fields if needed
            if BACKWARD_ROOT_PREFIX in name and "AccumulateGrad" not in name:
                if active_mode != Mode.BACKWARD:
                    # beginning of backward sequence
                    active_mode = Mode.BACKWARD
                    seq_end = start + duration
                    current_seq = new_event["Sequence number"]
                    current_root = new_event["event_id"]
                else:
                    # if we are in backward mode, extend the current backward sequence
                    if start + duration > seq_end:
                        # _active_mode remains BACKWARD
                        seq_end, current_seq = (
                            start + duration,
                            new_event["Sequence number"],
                        )
                        current_root = new_event["event_id"]
                new_event["mode"] = str(Mode.BACKWARD)
                self._group_events_by_sequence(new_event, grouped_by_seq_mode)
                continue
            else:
                if active_mode == Mode.BACKWARD and start > seq_end:
                    # end the sequence if current event is after the end of the backward sequence
                    active_mode, seq_end, current_seq, current_root = (
                        Mode.FORWARD,
                        -1,
                        -1,
                        -1,
                    )
                elif active_mode == Mode.BACKWARD:
                    # continue in backward mode, mark the event as a child of the root event
                    new_event["parent_id"] = current_root
                    new_event["mode"] = str(Mode.BACKWARD)
                    new_event["Sequence number"] = current_seq
                    external_map[new_event["External id"]] = (
                        current_seq,
                        new_event["event_id"],
                        new_event["mode"],
                    )
                    self._group_events_by_sequence(new_event, grouped_by_seq_mode)
                    continue

            assert active_mode == Mode.FORWARD
            if seq_end == -1:
                # beginning of forward sequence
                seq_end = start + duration
                current_seq = new_event["Sequence number"]
                current_root = new_event["event_id"]
            else:
                # extend the current forward sequence
                if start + duration > seq_end:
                    seq_end, current_seq = (
                        start + duration,
                        new_event["Sequence number"],
                    )
                    current_root = new_event["event_id"]
                else:
                    # if the current event is within the time window of the current forward sequence,
                    # mark it as a child of the root event
                    new_event["parent_id"] = current_root
                    new_event["Sequence number"] = current_seq

            external_map[new_event["External id"]] = (
                current_seq,
                new_event["event_id"],
                new_event["mode"],
            )
            if (
                new_event["Sequence number"] not in seq_module_map
                and new_event["nn_stack"]
            ):
                seq_module_map[new_event["Sequence number"]] = new_event["nn_stack"]
            self._group_events_by_sequence(new_event, grouped_by_seq_mode)

        self._merge_kernel_events(kernel_events, external_map, grouped_by_seq_mode)
        self._rename_keys_with_seq_module_map(seq_module_map, grouped_by_seq_mode)
        self._sort_and_build_nested_events(grouped_by_seq_mode)
        return grouped_by_seq_mode

    def _update_annotation_stack(self, name, stack, current_seq):
        if name.startswith("s:"):
            parts = name.split(":")
            if len(parts) > 2 and parts[1].isdigit():
                current_seq = -int(parts[1])
                stack.append(":".join(p for i, p in enumerate(parts) if i != 1))
            else:
                stack.append(name)
        elif name.startswith("f:") and stack:
            stack.pop()
        return current_seq

    def _build_cpu_op_event(self, e, annotation_stack, mode, seq_num):
        nn_stack = annotation_stack[-1][2:] if annotation_stack else ""
        return {
            "event_name": e["name"],
            "event_id": e["args"].get("Ev Idx", -1),
            "cat": e["cat"],
            "tid": e["tid"],
            "pid": e["pid"],
            "time_stamp": e["ts"],
            "end_time_stamp": e["ts"] + e.get("dur", 0),
            "Sequence number": e["args"].get("Sequence number", seq_num),
            "External id": e["args"].get("External id", -1),
            "mode": mode,
            "nn_stack": nn_stack,
            "parent_id": -1,
        }

    def _group_events_by_sequence(self, event, grouped):
        key, direction = event["Sequence number"], event["mode"]
        grouped[key][direction].append(event)

    def _merge_kernel_events(self, kernel_events, external_map, grouped):
        for k in kernel_events:
            eid = k["External id"]
            if eid in external_map:
                seq_num, parent_id, mode = external_map[eid]
                k["Sequence number"] = seq_num
                k["parent_id"] = parent_id
                k["mode"] = mode
                grouped[seq_num][mode].append(k)

    def _rename_keys_with_seq_module_map(self, seq_module_map, grouped):
        for seq_num in list(grouped.keys()):
            module_name = seq_module_map.get(seq_num, "Unknown")
            new_key = f"{seq_num}; Stack: {module_name}"
            new_key = new_key if len(new_key) <= 120 else new_key[:120]
            grouped[new_key] = grouped.pop(seq_num)

    def _sort_and_build_nested_events(self, grouped):
        for key, directions in grouped.items():
            for direction, events in directions.items():
                sorted_events = sorted(events, key=lambda x: x["time_stamp"])
                grouped[key][direction] = self._build_nested_structure(sorted_events)

    def _write_grouped_output(self, grouped_by_seq_mode, output_file):
        path_json = f"{self.processed_dir}/{output_file.replace('.yaml', '.json')}"
        with open(path_json, "wb") as f:
            f.write(orjson.dumps(grouped_by_seq_mode, option=orjson.OPT_INDENT_2))

    def _build_nested_structure(self, events):
        event_dict = {e["event_id"]: e for e in events}
        root_events = []
        for e in events:
            parent = e["parent_id"]
            if parent == -1:
                root_events.append(e)
            else:
                parent_event = event_dict.get(parent)
                if parent_event:
                    parent_event.setdefault("successors", []).append(e)

        def format_event(event):
            formatted = {event["cat"]: event["event_name"]}
            if event.get("successors"):
                formatted["successors"] = [
                    format_event(child) for child in event["successors"]
                ]
            return formatted

        return [format_event(event) for event in root_events]

    def parse(self):
        pytorch_trace_dir = f"{self.execution_dir}/tools/pytorch-trace"
        trace_files = pathlib.Path(pytorch_trace_dir).glob("*.pt.trace.json")
        for trace_file in trace_files:
            with open(trace_file, "r") as f:
                trace_data = json.load(f)
            if not trace_data or "traceEvents" not in trace_data:
                self.log.warning(
                    f"Unable to parse PyTorch trace data from {trace_file}"
                )
                continue
            output_file = f"{trace_file.stem}.callgraph.yaml"
            self.parseTraceData(trace_data, output_file)


class SysInfoParser(ProcessParser):
    def parse(self):
        omnihub_info_dir = f"{self.execution_dir}/sysinfo"
        data = {}
        for p in pathlib.Path(omnihub_info_dir).glob("*.json"):
            if p.name.startswith("amd_topology"):
                continue
            with open(p, "r") as f:
                rank_data = json.load(f)
                if "amd_firmware" in p.name:
                    data["amd_firmware"] = rank_data[0]
                elif "amd_static" in p.name and "driver" in rank_data[0]:
                    data["amd_driver"] = rank_data[0]["driver"]
                else:
                    data[p.stem] = rank_data

        if len(data) == 0:
            self.log.warning("Unable to parse sysinfo data")
            return

        with open(f"{self.processed_dir}/sysinfo.yaml", "w") as f:
            yaml.dump(data, f)


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
            "Start Timestamp": start,
            "End Timestamp": end,
            "Duration (s)": (end - start),
            "Rank mean duration (s)": statistics.mean(durations),
            "Rank min duration (s)": min(durations),
            "Rank max duration (s)": max(durations),
            "GPU energy (kWh)": energy,
        }

        with open(f"{self.processed_dir}/omnihub-monitor.yaml", "w") as f:
            yaml.dump(records, f)


# Omnistat range parser reads start and end timestamps from the omnihub monitor parser and calculates the average
# and maximum GPU utilization and memory usage over that time period.
# The parser reads from the Omnistat CSV file and writes the results to a YAML file.
class OmnistatRangeParser(ProcessParser):
    def __init__(self, execution_dir, name="omnistat"):
        super().__init__(execution_dir)
        self.variant_name = name

    def parse(self):
        # read the start and end timestamps from the output of the omnihub monitor parser
        # and use them to filter the omnistat data
        omnihub_monitor_file = f"{self.processed_dir}/omnihub-monitor.yaml"
        if not os.path.exists(omnihub_monitor_file):
            self.log.warning(
                f"Unable to find Omnihub monitor file: {omnihub_monitor_file}. Run the OmniHubMonitorParser first."
            )
            return

        with open(omnihub_monitor_file, "r") as f:
            data = yaml.safe_load(f)
        start_time = data["Start Timestamp"]
        end_time = data["End Timestamp"]

        omnistat_file = f"{self.execution_dir}/tools/{self.variant_name}/export.csv"
        if not os.path.exists(omnistat_file):
            self.log.warning(
                f"Unable to find Omnistat exported CSV file: {omnistat_file}"
            )
            return
        # The CSV file is expected to have a multi-index header with three levels.
        # Use the timestamp column as the index.
        # This structure is critical for parsing the data correctly. If the CSV format changes,
        # update the `header` parameter and ensure the rest of the code is compatible.
        df_data = pd.read_csv(omnistat_file, header=[0, 1, 2], index_col=0)

        # Convert the index (timestamp) to numeric (in seconds)
        df_data.index = pd.to_datetime(df_data.index).astype("int64") // 10**9

        # Filter rows based on the timestamp range
        df_filtered = df_data[
            (df_data.index >= start_time) & (df_data.index <= end_time)
        ]
        if df_filtered.empty:
            self.log.warning("No data available within the specified timestamp range.")
            return

        # df_data.columns have the following format:
        # ('metric', 'node_instance_name', 'gpu_index')
        # where metrics of interest can be one of the following, among others:
        # 'rocm_utilization_percentage', 'rocm_vram_used_percentage', 'rocm_average_socket_power_watts'
        metrics_map = {
            "rocm_utilization_percentage": "utilization",
            "rocm_vram_used_percentage": "memory",
            "rocm_average_socket_power_watts": "power",
        }
        result = {
            "Start Timestamp": start_time,
            "End Timestamp": end_time,
        }
        for metric, name in metrics_map.items():
            cols = [col for col in df_data.columns if col[0] == metric]
            if not cols:
                self.log.warning(f"Metric {metric} not found in Omnistat file")
                continue
            stats = df_filtered[cols]
            if name == "power":
                # compute energy in kWh: convert W*sec to Wh then to kWh
                time_diffs = df_filtered.index.to_series().diff().fillna(0)
                energy_Wh = (stats.mul(time_diffs, axis=0)).sum().sum() / 3600
                result["GPU energy (kWh)"] = float(energy_Wh) / 1000
            else:
                result[f"GPU mean {name} (%)"] = float(stats.mean(axis=1).mean())
                result[f"GPU max {name} (%)"] = float(stats.max(axis=1).max())

        with open(f"{self.processed_dir}/{self.variant_name}-range.yaml", "w") as f:
            yaml.dump(result, f)


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


class JobStatusParser(ProcessParser):
    def parse(self):
        with open(f"{self.execution_dir}/job-status.yaml", "r") as f:
            data = yaml.safe_load(f)
        duration_ms = data["execute_end_ms"] - data["execute_start_ms"]
        data["execute_duration_s"] = duration_ms / 1000
        with open(f"{self.processed_dir}/job-status.yaml", "w") as f:
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


class ReportCardParser(ProcessParser):
    def parse(self):
        # Define a mapping for each report field: target field -> (source file key, key in that file, default value)
        field_mapping = {
            "GPU max utilization (%)": (
                "omnistat-range",
                "GPU max utilization (%)",
                None,
            ),
            "GPU max memory (%)": ("omnistat-range", "GPU max memory (%)", None),
            "GPU mean utilization (%)": (
                "omnistat-range",
                "GPU mean utilization (%)",
                None,
            ),
            "GPU mean memory (%)": ("omnistat-range", "GPU mean memory (%)", None),
            "GPU energy (kWh)": ("omnihub-monitor", "GPU energy (kWh)", None),
            "Duration (s)": ("omnihub-monitor", "Duration (s)", None),
        }

        # Define the file paths for each report section
        file_mapping = {
            "job-status": f"{self.processed_dir}/job-status.yaml",
            "omnistat-range": f"{self.processed_dir}/omnistat-range.yaml",
            "omnihub-monitor": f"{self.processed_dir}/omnihub-monitor.yaml",
        }

        # Initialize data with default values per the mapping
        data = {field: default for field, (_, _, default) in field_mapping.items()}

        # Read job status file to get the exit code
        job_status_file = file_mapping["job-status"]
        exit_code = None
        with open(job_status_file, "r") as f:
            job_status_data = yaml.safe_load(f)
            exit_code = job_status_data.get("exit_code", None)

        # Only load the additional reports if the job was successful
        if exit_code == 0:
            loaded_files = {}
            for field, (section, key, default) in field_mapping.items():
                if section not in loaded_files:
                    file_path = file_mapping.get(section)
                    if file_path and os.path.exists(file_path):
                        with open(file_path, "r") as f:
                            loaded_files[section] = yaml.safe_load(f)
                    else:
                        loaded_files[section] = {}
                section_data = loaded_files.get(section, {})
                data[field] = section_data.get(key, default)

        # Save the parsed report card to a YAML file
        output_file = f"{self.processed_dir}/report-card.yaml"
        with open(output_file, "w") as f:
            yaml.dump(data, f)
