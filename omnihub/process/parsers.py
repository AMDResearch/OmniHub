import hashlib
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

    def load(self):
        pass

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
        app_config_file = pathlib.Path(f"{self.execution_dir}/app.yaml")
        if not app_config_file.exists():
            self.log.warning("Missing app configuration")
            return

        with open(app_config_file, "r") as f:
            data = yaml.safe_load(f)

        if not "entrypoint" in data:
            self.log.warning("Missing entrypoint field in job configuration")
            return

        # Mapping of entrypoints to application names. Entrypoints are the
        # only reliable way to know what application has been executed, which
        # is relevant to run the appropriate parser.
        app_names = {
            "applications/hf-finetune/finetune.py": "hf-finetune",
            "/app/vllm/benchmarks/benchmark_latency.py": "vllm-latency",
            "/app/vllm/benchmarks/benchmark_throughput.py": "vllm-throughput",
        }

        if not data["entrypoint"] in app_names:
            return

        # Assume application log parser is available if the `parse.py`
        # executable is present in the application directory.
        name = app_names[data["entrypoint"]]
        parser = f"applications/{name}/parse.py"
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


class RcclInfoParser(ProcessParser):
    def load(self):
        omnihub_monitor_file = f"{self.processed_dir}/omnihub-monitor.yaml"
        if not os.path.exists(omnihub_monitor_file):
            self.log.warning(
                f"Unable to find Omnihub monitor file: {omnihub_monitor_file}. Run the OmniHubMonitorParser first."
            )
            return

        with open(omnihub_monitor_file, "r") as f:
            data = yaml.safe_load(f)

        self.start_time = data["Start Timestamp"]
        self.end_time = data["End Timestamp"]

        # Check if the start and end times are valid
        if not isinstance(self.start_time, (int, float)) or not isinstance(
            self.end_time, (int, float)
        ):
            self.log.warning("Invalid start or end time in Omnihub monitor data")
            return
        if self.start_time >= self.end_time:
            self.log.warning("Start time must be less than end time")
            return

    def parse(self):
        # Create a histogram of the number of times each RCCL collective type is called for each datatype. Also collect the payload size and calculate the overall bandwidth.
        # Possible collective opType functions are "AllGather" "AllReduce" "Broadcast" "Recv" "Reduce" "ReduceScatter" "SendRecv" "Send"
        rccl_info_dir = f"{self.execution_dir}/tools/rccl-info"

        # The datatype is an enum (ncclDataType_t) from /opt/rocm/include/rccl/rccl.h. This maps the enum number to (datatype name, size in bytes)
        datatype_mapping = {
            "0": ("ncclInt8", 1),
            "1": ("ncclUint8", 1),
            "2": ("ncclInt32", 4),
            "3": ("ncclUint32", 4),
            "4": ("ncclInt64", 8),
            "5": ("ncclUint64", 8),
            "6": ("ncclFloat16", 2),
            "7": ("ncclFloat32", 4),
            "8": ("ncclFloat64", 8),
            "9": ("ncclBfloat16", 2),
            "10": ("ncclFloat8e4m3", 1),
            "11": ("ncclFloat8e5m2", 1),
        }

        total_byte_counts = {}
        # Generate per-rank data files with RCCL stats.
        rank_dirs = pathlib.Path(rccl_info_dir).iterdir()
        for rank_dir in rank_dirs:
            rank = rank_dir.name
            txt_files = list(rank_dir.glob("*.txt"))
            if not txt_files:
                self.log.warning(f"No RCCL info txt files found for rank {rank}")
                continue
            # there must be only one txt file per rank
            if len(txt_files) > 1:
                self.log.warning(
                    f"Multiple RCCL info txt files found for rank {rank}, using the first one."
                )
            info_file = txt_files[0]

            counts = defaultdict(lambda: defaultdict(int))
            byte_counts = defaultdict(lambda: defaultdict(int))
            with open(info_file, "r") as f:
                for line in f:
                    # Parse the line to extract opType and datatype fields. Format is like below:
                    # timestamp rank [1] NCCL INFO opType: opCount <opCount> sendbuff <sendBuff> recvbuff <recvBuff> count <count> datatype <datatype> op <op> root <root> comm <comm> [nranks=<nranks>] stream <stream> task <task> globalrank <globalrank>
                    timestamp_str = r"^(\d+\.\d+)"
                    local_rank_str = r"\s+\S+\s+\[(\d+)\]"
                    info = r"\s+NCCL INFO\s+"
                    op_type = r"(\w+):"
                    op_count = r"\s+opCount\s+\S+"
                    sendbuff = r"\s+sendbuff\s+\S+"
                    recvbuff = r"\s+recvbuff\s+\S+"
                    count = r"\s+count\s+(\d+)"
                    datatype = r"\s+datatype\s+(\d+)"
                    op_root_comm = r"\s+op\s+\S+\s+root\s+\S+\s+comm\s+\S+"
                    optional_nranks = r"(?:\s+\[nranks=\S+\])?"
                    stream_task_globalrank = (
                        r"\s+stream\s+\S+\s+task\s+\S+\s+globalrank\s+\S+"
                    )

                    pattern = (
                        timestamp_str
                        + local_rank_str
                        + info
                        + op_type
                        + op_count
                        + sendbuff
                        + recvbuff
                        + count
                        + datatype
                        + op_root_comm
                        + optional_nranks
                        + stream_task_globalrank
                    )

                    match = re.search(pattern, line)
                    if match:
                        timestamp, local_rank, opType, dtype_count, dtype_id = (
                            match.group(1, 2, 3, 4, 5)
                        )
                        timestamp = float(timestamp)
                        if not (self.start_time <= timestamp <= self.end_time):
                            continue
                        dtype_name, dtype_size = datatype_mapping.get(
                            dtype_id, ("unknown", 0)
                        )
                        counts[opType][dtype_name] += 1
                        byte_counts[opType][dtype_name] += int(dtype_size) * int(
                            dtype_count
                        )

            output_filename = (
                f"{self.processed_dir}/rccl-info-collectives.histogram.{rank}.yaml"
            )
            records_counts = {
                f"{op}#{dtype}": count
                for op, dtypes in counts.items()
                for dtype, count in dtypes.items()
            }
            with open(output_filename, "w") as f:
                yaml.dump(records_counts, f)

            byte_counts_section = f"byte_counts.{rank}"
            output_filename = (
                f"{self.processed_dir}/rccl-info-collectives.bytes.{rank}.yaml"
            )
            total_byte_counts[byte_counts_section] = {
                f"{op}#{dtype}": value
                for op, dtypes in byte_counts.items()
                for dtype, value in dtypes.items()
            }
            with open(output_filename, "w") as f:
                yaml.dump(total_byte_counts[byte_counts_section], f)

        # Calculate bandwidth in MB/s from total bytes and duration
        total_byte_counts_df = pd.DataFrame(total_byte_counts).fillna(0)
        total_byte_counts_float = float(total_byte_counts_df.sum(axis=1).sum())

        records = {}
        duration = self.end_time - self.start_time
        records["RCCL Collectives Payload (MB)"] = total_byte_counts_float / (1024**2)
        records["RCCL Collectives Bandwidth (MiB/s)"] = (
            total_byte_counts_float / (1024**2) / duration
        )

        # Write the summarized data to a YAML file
        with open(f"{self.processed_dir}/rccl-info-collectives.yaml", "w") as f:
            yaml.dump(records, f)


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
            "Rank Mean Duration (s)": statistics.mean(durations),
            "Rank Min Duration (s)": min(durations),
            "Rank Max Duration (s)": max(durations),
            "GPU Energy (kWh)": energy,
        }

        with open(f"{self.processed_dir}/omnihub-monitor.yaml", "w") as f:
            yaml.dump(records, f)


class OmnistatParser(ProcessParser):
    def __init__(self, execution_dir, name="omnistat"):
        super().__init__(execution_dir)
        self.variant_name = name
        self.output_file = f"{self.processed_dir}/omnistat.yaml"

        # Optional start/end times to parse only a specific time range.
        # If this range is not defined, use the entire CSV files.
        self.start_time = None
        self.end_time = None

    def parse_csv(self, csv_file, metrics, levels):
        """
        Parse Omnistat data from exported CSV file.

        Args:
            csv_file (string): File name.
            metrics (list(str)): List of metrics
            levels (int): Number of levels in the multi-index header hierarchy.

        Returns:
            pandas.DataFrame: Data frame with a subset of the data in the CSV
            file, including only the given time range and metrics.
        """
        csv_path = f"{self.execution_dir}/tools/{self.variant_name}/{csv_file}"
        if not os.path.exists(csv_path):
            self.log.warning(f"Unable to find Omnistat exported CSV file: {csv_path}")
            return None

        # The CSV file is expected to have a multi-index header with the given
        # number of levels.
        df = pd.read_csv(csv_path, header=list(range(levels)), index_col=0)

        # Convert the index (timestamp) to numeric (in seconds)
        df.index = pd.to_datetime(df.index).astype("int64") / 10**9

        # Filter rows based on the timestamp range
        if self.start_time != None and self.end_time != None:
            df = df[(df.index >= self.start_time) & (df.index <= self.end_time)]

        return df[metrics]

    def parse_rocm_data(self):
        result = {}
        metrics = {
            "rocm_utilization_percentage": "GPU Utilization (%)",
            "rocm_vram_used_percentage": "GPU Memory Utilization (%)",
            "rocm_average_socket_power_watts": "GPU Power (W)",
        }
        df = self.parse_csv("omnistat-rocm.csv", metrics.keys(), levels=3)
        if df is None or df.empty:
            return result

        for metric, name in metrics.items():
            df_metric = df[metric]
            result[f"{name} Mean"] = float(df_metric.mean(axis=1).mean())
            result[f"{name} Max"] = float(df_metric.max(axis=1).max())

        # Compute energy in kWh: convert W*sec to Wh then to kWh.
        power_df = df["rocm_average_socket_power_watts"]
        time_diffs = power_df.index.to_series().diff().fillna(0)
        energy = (power_df.mul(time_diffs, axis=0)).sum().sum() / 3600
        result["GPU Energy (kWh)"] = float(energy) / 1000

        return result

    def parse_network_data(self):
        result = {}
        metrics = {
            "omnistat_network_rx_bytes": "Network Rx {} (MiB/s) Mean",
            "omnistat_network_tx_bytes": "Network Tx {} (MiB/s) Mean",
        }

        df = self.parse_csv("omnistat-network.csv", metrics.keys(), levels=4)
        if df is None or df.empty:
            return result

        start_time = df.index[0]
        end_time = df.index[-1]
        duration = float(end_time - start_time)

        if duration <= 0:
            self.log.warning("Insufficient data in CSV file.")
            return result

        df = df.iloc[-1] - df.iloc[0]
        for metric, name in metrics.items():
            df_all = df[metric]
            name_all = name.format("All")
            result[name_all] = float(df_all.sum()) / (1024**2) / duration

            df_net = df_all.loc[pd.IndexSlice[:, ["net"]]]
            name_net = name.format("Ethernet")
            result[name_net] = float(df_net.sum()) / (1024**2) / duration

            device_classes = set(df_all.index.get_level_values("device_class"))
            if "infiniband" in device_classes:
                df_ib = df_all.loc[pd.IndexSlice[:, ["infiniband"]]]
                name_ib = name.format("InfiniBand")
                result[name_ib] = float(df_ib.sum()) / (1024**2) / duration

        return result

    def parse_rocprofiler_data(self):
        result = {}

        metrics = ["omnistat_rocprofiler"]
        df = self.parse_csv("omnistat-rocprofiler.csv", metrics, levels=4)
        if df is None or df.empty:
            return result

        start_time = df.index[0]
        end_time = df.index[-1]
        duration = float(end_time - start_time)

        if duration <= 0:
            self.log.warning("Insufficient data in CSV file.")
            return result

        df = df.iloc[-1] - df.iloc[0]
        df = df.groupby(level="counter").sum()
        for counter, value in df.items():
            result[counter] = value

        return result

    def parse(self):
        result = {}
        result.update(self.parse_rocm_data())
        result.update(self.parse_network_data())
        if self.variant_name.startswith("omnistat-rocprofiler"):
            result.update(self.parse_rocprofiler_data())
        with open(self.output_file, "w") as f:
            yaml.dump(result, f)


# Alternative Omnistat parser that generates the same data as the default
# Omnistat parser over a specific time range. The time range is extracted from
# the timestamps available in the Omnihub monitor parser, which is expected to
# be executed first.
class OmnistatRangeParser(OmnistatParser):
    def __init__(self, execution_dir, name="omnistat"):
        super().__init__(execution_dir, name)
        self.output_file = f"{self.processed_dir}/omnistat-range.yaml"

    def load(self):
        omnihub_monitor_file = f"{self.processed_dir}/omnihub-monitor.yaml"
        if not os.path.exists(omnihub_monitor_file):
            self.log.warning(
                f"Unable to find Omnihub monitor file: {omnihub_monitor_file}. Run the OmniHubMonitorParser first."
            )
            return

        with open(omnihub_monitor_file, "r") as f:
            data = yaml.safe_load(f)

        self.start_time = data["Start Timestamp"]
        self.end_time = data["End Timestamp"]


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
            "GPU Utilization (%) Max": max([i["util-max"] for i in gpu_data]),
            "GPU Utilization (%) Mean": statistics.mean(
                [i["util-mean"] for i in gpu_data]
            ),
            "GPU Memory Utilization (%) Max": max([i["mem-max"] for i in gpu_data]),
            "GPU Memory Utilization (%) Mean": statistics.mean(
                [i["mem-mean"] for i in gpu_data]
            ),
        }

        energy_pattern = re.compile(
            r"Approximate Total GPU Energy Consumed = ([0-9]*\.[0-9]+|[0-9]+) kWh"
        )
        for line in report:
            match = re.match(energy_pattern, line)
            if match:
                records["GPU Energy (kWh)"] = float(match.group(1))

        # Assume executions only have a single Omnistat variant, and so there
        # is a single report to process, and it's always stored in the same
        # omnistat.yaml file.
        with open(f"{self.processed_dir}/omnistat-report.yaml", "w") as f:
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
            "GPU Utilization (%) Max": (
                "omnistat-range",
                "GPU Utilization (%) Max",
                None,
            ),
            "GPU Memory Utilization (%) Max": (
                "omnistat-range",
                "GPU Memory Utilization (%) Max",
                None,
            ),
            "GPU Utilization (%) Mean": (
                "omnistat-range",
                "GPU Utilization (%) Mean",
                None,
            ),
            "GPU Memory Utilization (%) Mean": (
                "omnistat-range",
                "GPU Memory Utilization (%) Mean",
                None,
            ),
            "Network Transmit Bandwidth (MiB/s) Mean": (
                "omnistat-range",
                "Network Tx All (MiB/s) Mean",
                None,
            ),
            "Network Receive Bandwidth (MiB/s) Mean": (
                "omnistat-range",
                "Network Rx All (MiB/s) Mean",
                None,
            ),
            "RCCL Collectives Bandwidth (MiB/s)": (
                "rccl-info-collectives",
                "RCCL Collectives Bandwidth (MiB/s)",
                None,
            ),
            "GPU Energy (kWh)": ("omnihub-monitor", "GPU Energy (kWh)", None),
            "GPU Energy omnistat (kWh)": ("omnistat-range", "GPU Energy (kWh)", None),
            "Duration (s)": ("omnihub-monitor", "Duration (s)", None),
        }

        # Define the file paths for each report section
        file_mapping = {
            "job-status": f"{self.processed_dir}/job-status.yaml",
            "omnistat-range": f"{self.processed_dir}/omnistat-range.yaml",
            "omnihub-monitor": f"{self.processed_dir}/omnihub-monitor.yaml",
            "rccl-info-collectives": f"{self.processed_dir}/rccl-info-collectives.yaml",
        }

        # Ensure all required files exist; if any file is missing, return early.
        for _, file_path in file_mapping.items():
            if not os.path.exists(file_path):
                self.log.warning(f"Missing required file: {file_path}")
                return

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

        # If the energy from omnihub-monitor is not available, use the one from omnistat-range
        if data["GPU Energy (kWh)"] == 0:
            data["GPU Energy (kWh)"] = data["GPU Energy omnistat (kWh)"]
        # Remove the temporary field used for fallback
        data.pop("GPU Energy omnistat (kWh)", None)

        # Save the parsed report card to a YAML file
        output_file = f"{self.processed_dir}/report-card.yaml"
        with open(output_file, "w") as f:
            yaml.dump(data, f)


# Create a hash of the execution directory to uniquely identify groups of related executions.
class HashParser(ProcessParser):
    def parse(self):
        # read the dictionaries from app.yaml and job.yaml, create a hash
        app_file = f"{self.execution_dir}/app.yaml"
        job_file = f"{self.execution_dir}/job.yaml"
        if not os.path.exists(app_file) or not os.path.exists(job_file):
            self.log.warning(
                f"Unable to find app.yaml or job.yaml in {self.execution_dir}. Skipping hash generation."
            )
            return
        with open(app_file, "r") as f:
            app_data = yaml.safe_load(f)
        with open(job_file, "r") as f:
            job_data = yaml.safe_load(f)
        job_data = job_data.get("job", {})
        if not app_data or not job_data:
            self.log.warning("App or job data is empty. Skipping hash generation.")
            return
        # Skip specific keys in job data. Note: we are skipping "cluster" and "container-platform" assuming that
        # different clusters or containers are treated as different executions, and so we don't want to include them in
        # the hash.
        keys_to_skip = [
            "app-config",
            "head-address",
            "head-node",
            "head-port",
            "id",
            "models-directory",
            "nodes",
            "omnihub-directory",
            "time-limit-seconds",
            "timestamp",
            "tools",
            "user",
        ]
        for key in keys_to_skip:
            job_data.pop(key, None)
        # Create a combined dictionary of the app and job data
        combined_data = {
            "app": app_data,
            "job": job_data,
        }
        # Create a hash of the combined data
        combined_str = yaml.dump(combined_data, sort_keys=True)
        execution_hash = hashlib.sha256(combined_str.encode()).hexdigest()

        execution_hash_record = {
            "execution_hash": execution_hash,
        }
        # Save the hash to a YAML file in the processed directory as a dictionary
        hash_file = f"{self.processed_dir}/execution_hash.yaml"
        with open(hash_file, "w") as f:
            yaml.dump(execution_hash_record, f)
