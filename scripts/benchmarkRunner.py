#! /usr/bin/env python3

import argparse
def getCommandLineArguments():
    parser = argparse.ArgumentParser(description='This python script runs a single benchmark, \
            possibly with a prefixing submit command like \"perf stat --\". \
            The script creates a new output directory in the current working directory, \
            copy the benchmark files there, and then pre_run, run, and post_run the benchmark. \
            Finally, the script deletes large files (> 1MB) residing in the output directory.')
    parser.add_argument('-n', '--num_threads', type=int, default=4,
            help='uses this number of threads (for multi-threaded benchmark)')
    parser.add_argument('--repeat', type=str, default=None,
        help='run only one repeat into output_dir/<repeat> (e.g., repeat1).')
    parser.add_argument('-s', '--submit_command', type=str, default='',
            help='a command that will prefix running the benchmark, e.g., "perf stat --".')
    parser.add_argument('-c', '--clean_threshold', type=int, default=1024*1024,
            help='delete files larger than this size (in bytes) after the benchmark runs')
    parser.add_argument('-x', '--exclude_files', type=str, nargs='*', default=[],
            help='do not remove these files')
    parser.add_argument('-p', '--prefix', type=str, default=None,
            help='a command line to be used as a prefix for the submit command')
    parser.add_argument('-f', '--force', action='store_true', default=False,
            help='run the benchmark anyway even if the output directory already exists')
    parser.add_argument('-pre', '--pre_run', action='store_true', default=False,
            help='run the pre_run script')
    parser.add_argument('-post', '--post_run', action='store_true', default=True,
            help='run the post_run script')
    parser.add_argument('-bench', '--benchmark_dir', type=str, required=True,
            help='the benchmark directory, must contain three bash scripts: pre_run.sh, run.sh, and post_run.sh')
    parser.add_argument('-run', '--run_dir', type=str, required=True,
            help='the directory which will be created for running the benchmark for all experiments and layouts')
    parser.add_argument('-out', '--output_dir', type=str, required=True,
            help='the output directory which will be created for saving the output of the benchmark run')
    parser.add_argument('-l', '--loop_until', type=int, default=None,
            help='run the benchmark repeatedly until LOOP_UNTIL seconds have passed')

    args = parser.parse_args()
    return args

from benchmarkCore import BenchmarkRun
from pathlib import Path
import time
if __name__ == "__main__":
    args = getCommandLineArguments()

    # then replace repeated_runs construction with:
    if args.loop_until is not None:
        assert args.repeat is not None, "When using --loop_until, you must also specify --repeat to name the output directory."
        repeated_runs = [BenchmarkRun(args.benchmark_dir, args.run_dir, args.output_dir + '/' + args.repeat)]
    elif args.repeat is not None:
        repeated_runs = [BenchmarkRun(args.benchmark_dir, args.run_dir, args.output_dir + '/' + args.repeat)]
    
    existing_repeat_dirs = 0
    for run in repeated_runs:
        if run.doesOutputDirectoryExist():
            existing_repeat_dirs += 1
    if existing_repeat_dirs == len(repeated_runs) and not args.force:
        print(f'Skipping the run because the output directory [{args.output_dir}] already exists.')
        print('You can use the \'-f\' flag to suppress this message and run the benchmark anyway.')
        exit(0)

    # replace prerun with warmup, which runs the benchmark before other runs

    run_cmd = args.submit_command
    if args.prefix is not None:
        run_cmd = f'{args.prefix} {args.submit_command}'
    for run in repeated_runs: # run for each repeat
        if args.pre_run:
            print(f'start pre-running...')
            run.prerun()
        print('================================================')
        print(f'start producing:\n\t{run._output_dir}')


        if args.loop_until is not None:
            run.run_loop_until(args.num_threads, run_cmd, args.loop_until)
        else:
            p = run.run(args.num_threads, run_cmd)
            p.check_returncode()

            if args.post_run:
                print(f'start post-running...')
                run.postrun()

        # move output files to out_dir
        #   (should be done after postrun to allow processing output files
        #    in the run_dir before moving them to the output_dir)
        run.move_files_to_output_dir()

        # clean out_dir
        run.clean_output_dir(args.clean_threshold, args.exclude_files)

        print('================================================')




