#! /usr/bin/env python3

import argparse
import time
import subprocess
import shutil
import shlex

import signal
import psutil
import os
import os.path
import sys
import glob

from pathlib import Path

# try to kill all subprocesses if this script is killed by a signal from the user
def killAllSubprocesses(signum, frame):
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        print(f"Killing child process {child.pid}")
        try:
            os.kill(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass # the child process may have terminated already
    sys.exit(f"Exiting due to a {signal.Signals(signum).name} signal")

signal.signal(signal.SIGINT, killAllSubprocesses)
signal.signal(signal.SIGTERM, killAllSubprocesses)


class BenchmarkRun:
    def __init__(self, benchmark_dir: str, run_dir: str, output_dir: str):
        self._benchmark_dir = Path(benchmark_dir).absolute()
        self._assertBenchmarkIsValid()

        self._run_dir = Path(run_dir).absolute()
        self._createNewRunDirectory()

        self._output_dir = Path(output_dir).absolute()
        self._createNewOutputDirectory()

        self._benchmark_files = set(os.listdir(self._benchmark_dir))

        log_file_name = self._output_dir / 'benchmark.log'
        self._log_file = open(log_file_name, 'w+')

    def __del__(self):
        if hasattr(self, "_log_file"):
            self._log_file.close()

    def _assertBenchmarkIsValid(self):
        if not self._benchmark_dir.exists():
            sys.exit(f'Error: the benchmark {self._benchmark_dir} was not found.\nDid you spell it correctly?')

    def _createNewRunDirectory(self):
        run_script = self._run_dir / 'run.sh'
        if not self._run_dir.exists() or not run_script.exists():
            print('******************************************************')
            print(f'*** copy benchmark files to the run directory ***')
            print(f'\t{self._benchmark_dir} --> {self._run_dir}')
            print('******************************************************')
            # symlinks are copied as symlinks with symlinks=True
            shutil.copytree(self._benchmark_dir, self._run_dir, dirs_exist_ok=True, symlinks=True)


    def _createNewOutputDirectory(self):
        if self._output_dir.exists():
            print(f'output directory {self._output_dir} already exists')
        else:
            print(f'creating a new output directory\n\t{self._output_dir}')
            self._output_dir.mkdir(parents=True, exist_ok=True)

    def doesOutputDirectoryExist(self):
        # True if it exists, it's a dir, and it contains any items
        perf_out_exists = False
        if self._output_dir.exists() and self._output_dir.is_dir():
            perf_out_exists = any((f.name == 'perf.out' or f.name == 'perf.data') and f.is_file() for f in self._output_dir.iterdir())
        return perf_out_exists

    def doesRunDirectoryExist(self):
        return self._run_dir.exists()

    # prerun is required, for example, to read input files into the page-cache before run() is invoked
    def prerun(self):
        print(f'{self._benchmark_dir}: prerunning')
        os.chdir(self._run_dir)
        pre_run_filename = glob.glob('pre*run.sh')
        if pre_run_filename == []:
            pre_run_filename = 'warmup.sh'
        else:
            pre_run_filename = pre_run_filename[0]
        subprocess.run(pre_run_filename, stdout=self._log_file, stderr=self._log_file, check=True)

    def run(self, num_threads: int, submit_command: str):
        print(f'{self._benchmark_dir}: running\n\t{submit_command} ./run.sh')

        # override the values already in the environment
        environment_variables = os.environ.copy()
        environment_variables.update(
                {
                    "OMP_NUM_THREADS"       : str(num_threads),
                    "OMP_THREAD_LIMIT"      : str(num_threads),
                    "OMP_PLACES"            : "cores",
                    "OMP_PROC_BIND"         : "true",
                    "OMP_SCHEDULE"          : "static",
                    "MOSMODEL_RUN_OUT_DIR"  : str(self._output_dir)
                 })

        # change dir to run_dir
        os.chdir(self._run_dir)

        # start running the benchmark
        p = subprocess.run(shlex.split(submit_command + ' ./run.sh'),
                env=environment_variables, stdout=self._log_file, stderr=self._log_file)

        return p

    def async_run(self, num_threads, submit_command):
        print('running the benchmark ' + self._benchmark_dir + '...')
        print('the full submit command is:\n\t' + submit_command + ' ./run.sh')
        environment_variables = {"OMP_NUM_THREADS": str(num_threads),
                "OMP_THREAD_LIMIT": str(num_threads)}
        environment_variables.update(os.environ)
        os.chdir(self._run_dir)
        self._async_process = subprocess.Popen(shlex.split(submit_command + ' ./run.sh'),
                stdout=self._log_file, stderr=self._log_file, env=environment_variables)

    def async_wait(self):
        print('waiting for the run to complete...')
        if not self._async_process:
            sys.exit(f'Error: there is no process running asynchronously!')

        self._async_process.wait()
        if self._async_process.returncode != 0:
            raise subprocess.CalledProcessError(self._async_process.returncode, ' '.join(self._async_process.args))

    # postrun is required, for example, to validate the run() outputs
    def postrun(self):
        print(f'{self._benchmark_dir}: postrunning')
        os.chdir(self._run_dir)
        post_run_filename = glob.glob('post*run.sh')
        if post_run_filename == []:
            post_run_filename = './validate.sh'
        else:
            post_run_filename = './' + post_run_filename[0]
        # sleep a bit to let the filesystem recover before running postrun.sh
        time.sleep(5)  # seconds
        subprocess.run(post_run_filename, stdout=self._log_file, stderr=self._log_file, check=True)

    def move_files_to_output_dir(self):
        # get updated list with all files in the run_dir
        post_run_files = set(os.listdir(self._run_dir))
        # get the list of all new files
        new_files = post_run_files - self._benchmark_files
        # Move only the new files to the output_dir
        for item in new_files:
            src_path = self._run_dir / item
            # Check if it's a file (or do similar logic for directories)
            if src_path.is_file():
                # override the file if it already exists in the output_dir
                dest_path = self._output_dir / item
                shutil.move(src_path, dest_path)
                print(f"Moved: {item} -> {dest_path}")

    def clean_dir(self, dir_path, threshold: int = 1024*1024, exclude_files: list = []):
        os.chdir(dir_path)
        for root, dirs, files in os.walk('./'):
            for name in files:
                file_path = os.path.join(root, name)
                # remove files larger than threshold (default is 1MB)
                if (not os.path.islink(file_path)) and (os.path.getsize(file_path) > threshold) and (name not in exclude_files):
                    os.remove(file_path)
        # sync to clean all pending I/O activity
        os.sync()

    def clean_output_dir(self, threshold: int = 1024*1024, exclude_files: list = []):
        print(f'{self._benchmark_dir}: cleaning large files from the output directory')
        self.clean_dir(self._output_dir, threshold, exclude_files)

    def clean_run_dir(self, threshold: int = 1024*1024, exclude_files: list = []):
        print(f'{self._benchmark_dir}: cleaning large files from the run directory')
        self.clean_dir(self._run_dir, threshold, exclude_files)

    def clean(self, threshold: int = 1024*1024, exclude_files: list = []):
        self.clean_run_dir(threshold, exclude_files)
        self.clean_output_dir(threshold, exclude_files)

def getCommandLineArguments():
    parser = argparse.ArgumentParser(description='This python script runs a single benchmark, \
            possibly with a prefixing submit command like \"perf stat --\". \
            The script creates a new output directory in the current working directory, \
            copy the benchmark files there, and then invoke prerun.sh, run.sh, and postrun.sh. \
            Finally, the script deletes large files (> 1MB) residing in the output directory.')
    parser.add_argument('-n', '--num_threads', type=int, default=4,
            help='use this number of threads (for multi-threaded benchmarks)')
    parser.add_argument('-s', '--submit_command', type=str, default='',
            help='prefix the benchmark run with this command (e.g., "perf stat --")')
    parser.add_argument('-c', '--clean_threshold', type=int, default=1024*1024,
            help='delete files larger than this size (in bytes) after the benchmark runs')
    parser.add_argument('-x', '--exclude_files', type=str, nargs='*', default=[],
            help='do not delete large files whose names appear in this list')
    parser.add_argument('-l', '--loop_until', type=int, default=None,
            help='run the benchmark repeatedly until LOOP_UNTIL seconds have passed')
    parser.add_argument('-t', '--timeout', type=int, default=None,
            help='timeout the benchmark run to TIMEOUT seconds')
    parser.add_argument('benchmark_dir', type=str, help='the benchmark directory, must contain three \
            bash scripts: pre_run.sh, run.sh, and post_run.sh')
    parser.add_argument('output_dir', type=str, help='the output directory which will be created for \
            running the benchmark on a clean slate')

    args = parser.parse_args()

    if args.timeout and args.loop_until:
        parser.error('only one of --timeout or --loop_until can be defined')
    if args.timeout and args.timeout <= 0:
        parser.error('timeout must be a positive integer')
    if args.loop_until and args.loop_until <= 0:
        parser.error('loop_until must be a positive integer')

    return args


if __name__ == "__main__":
    cwd = os.getcwd()
    args = getCommandLineArguments()
    os.chdir(cwd)
    run = BenchmarkRun(args.benchmark_dir, args.output_dir, args.output_dir)
    if not run.doesOutputDirectoryExist(): # skip existing directories
        run.prerun()
        if args.timeout:
            timeout_command = f'timeout {args.timeout}'
            p = run.run(args.num_threads, args.submit_command+' '+timeout_command)
            if p.returncode == 0: # the run ended before the timeout
                run.postrun()
        elif args.loop_until:
            loop_forever = './loopForever.sh'
            timeout_command = f'timeout {args.loop_until} {loop_forever}'
            run.run(args.num_threads, args.submit_command+' '+timeout_command)
            # don't check the exit status of run() because it was interrupted by timeout
            # don't call postrun() because we cannot validate a run that was interrupted by timeout
        else:
            p = run.run(args.num_threads, args.submit_command)
            p.check_returncode()
            run.postrun()
        run.clean(args.clean_threshold, args.exclude_files)

