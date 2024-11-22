import json
import logging
import os
import pathlib
import re
import statistics
import subprocess
from abc import ABC, abstractmethod

import pandas
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


class DefaultMonitorParser(ProcessParser):
    def parse(self):
        default_monitor_dir = f"{self.execution_dir}/tools/default"
        data = []
        for p in pathlib.Path(default_monitor_dir).glob("*.json"):
            with open(p, "r") as f:
                rank_data = json.load(f)
            data.append(rank_data)

        if len(data) == 0:
            self.log.warning("Unable to parse default monitor data")
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

        with open(f"{self.processed_dir}/default-monitor.yaml", "w") as f:
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
