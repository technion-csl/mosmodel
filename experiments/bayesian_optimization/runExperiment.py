#!/usr/bin/env python3
# import cProfile
import pandas as pd
from skopt import gp_minimize
from skopt.space import Integer, Space
from skopt.utils import use_named_args
import itertools
from numpy.polynomial.chebyshev import chebgauss
import numpy as np
from bitarray import bitarray
import subprocess
import math
import os, sys

curr_file_dir = os.path.dirname(os.path.abspath(__file__))
experiments_root_dir = os.path.join(curr_file_dir, '..')
sys.path.append(experiments_root_dir)
from Utils.utils import Utils

class BayesianExperiment:
    # based on https://arxiv.org/pdf/1807.02811.pdf
    MAX_DIMENSIONS = 20
    DEFAULT_HUGEPAGE_SIZE = 1 << 21 # 2MB 0

    def __init__(self,
                 memory_footprint_file, pebs_mem_bins_file,
                 collect_reults_cmd, results_file,
                 run_experiment_cmd, exp_root_dir,
                 num_layouts) -> None:
        self.last_layout_num = 0
        self.collect_reults_cmd = collect_reults_cmd
        self.results_file = results_file
        self.memory_footprint_file = memory_footprint_file
        self.pebs_mem_bins_file = pebs_mem_bins_file
        self.run_experiment_cmd = run_experiment_cmd
        self.exp_root_dir = exp_root_dir
        self.num_layouts = num_layouts
        self.prepare_space()

    def prepare_space(self):
        # read memory-footprints
        self.footprint_df = pd.read_csv(self.memory_footprint_file)
        self.mmap_footprint = self.footprint_df['anon-mmap-max'][0]
        self.brk_footprint = self.footprint_df['brk-max'][0]
        self.memory_footprint = self.brk_footprint

        self.hugepage_size = BayesianExperiment.DEFAULT_HUGEPAGE_SIZE
        self.num_hugepages = math.ceil(self.memory_footprint / self.hugepage_size) # bit vector length
        self.num_default_hugepages = math.ceil(self.memory_footprint / BayesianExperiment.DEFAULT_HUGEPAGE_SIZE)

        self.dimension_size_in_bits = 64 #sys.getsizeof(int)
        self.dimension_capacity = 2**self.dimension_size_in_bits
        # the num_dimensions is calculated for (num_hugepages + 1) because
        # an additional bit may be required when converting a binary number to gray code
        self.num_dimensions = math.ceil((self.num_hugepages + 1) / self.dimension_size_in_bits)
        if self.num_dimensions > BayesianExperiment.MAX_DIMENSIONS:
            self.num_dimensions = BayesianExperiment.MAX_DIMENSIONS
            # length(gray_code) = length(bit_vector) - 1
            self.max_num_hugepages = (BayesianExperiment.MAX_DIMENSIONS * self.dimension_size_in_bits) - 1
            self.hugepage_size = Utils.round_up(
                math.ceil(self.memory_footprint / self.max_num_hugepages),
                BayesianExperiment.DEFAULT_HUGEPAGE_SIZE)
            self.num_hugepages = math.ceil(self.memory_footprint / self.hugepage_size)
        # update num_dimensions and layout_bit_vector_length in case we exceeded the MAX_DIMESNIONS
        self.layout_bit_vector_length = self.num_hugepages
        self.gray_layout_bit_vector_length = self.layout_bit_vector_length + 1
        self.num_dimensions = math.ceil(self.gray_layout_bit_vector_length / self.dimension_size_in_bits)
        self.hugepages_in_compressed_hugepage = self.hugepage_size // BayesianExperiment.DEFAULT_HUGEPAGE_SIZE
        # Define the search space
        self.dimension_min_val = 0
        self.dimension_max_val = self.dimension_capacity - 1
        self.last_dimension_size_in_bits = self.gray_layout_bit_vector_length % self.dimension_size_in_bits
        self.last_dimension_max_val = 2**self.last_dimension_size_in_bits
        self.dimensions = [Integer(self.dimension_min_val, self.dimension_max_val, name=f'mem_region_{i}') for i in range(self.num_dimensions - 1)]
        self.dimensions += [Integer(self.dimension_min_val, self.last_dimension_max_val, name=f'mem_region_{self.num_dimensions-1}')]

        # round up the memory footprint to match the new boundaries of the new hugepage-size
        self.memory_footprint = (self.num_hugepages + 1) * self.hugepage_size
        self.brk_footprint = self.memory_footprint

        if self.pebs_mem_bins_file is None:
            print('pebs_mem_bins_file argument is missing, skipping loading PEBS results...')
            self.pebs_df = None
            self.total_misses = None
        else:
            self.pebs_df = Utils.load_pebs(self.pebs_mem_bins_file, True)
            self.total_misses = self.pebs_df['NUM_ACCESSES'].sum()

        if False:
            print(f'self.layout_bit_vector_length={self.layout_bit_vector_length}')
            print(f'self.gray_layout_bit_vector_length={self.gray_layout_bit_vector_length}')
            print(f'self.num_dimensions={self.num_dimensions}')
            print(f'self.hugepages_in_compressed_hugepage={self.hugepages_in_compressed_hugepage}')
            print(f'self.last_dimension_size_in_bits={self.last_dimension_size_in_bits}')
            print(f'self.num_hugepages={self.num_hugepages}')
            print(f'self.hugepage_size={self.hugepage_size}')
            print(f'self.memory_footprint={self.memory_footprint}')

    def run_command(command, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Run the command
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()

        # Get the output and error messages
        output = output.decode('utf-8')
        error = error.decode('utf-8')

        # Check the return code
        return_code = process.returncode

        output_log = f'{out_dir}/benchmark.log'
        error_log = f'{out_dir}/benchmark.log'
        with open(output_log, 'w+') as out:
            out.write(output)
            out.write('============================================')
            out.write(f'the process exited with status: {return_code}')
            out.write('============================================')
        with open(error_log, 'w+') as err:
            err.write(error)
            err.write('============================================')
            err.write(f'the process exited with status: {return_code}')
            err.write('============================================')
        if return_code != 0:
            # Print the output and error
            print('============================================')
            print(f'Failed to run the following command with exit code: {return_code}')
            print(f'Command line: {command}')
            print('Output:', output)
            print('Error:', error)
            print('Return code:', return_code)
            print('============================================')

        return return_code

    def collect_results(collect_reults_cmd, results_file):
        print(f'** collecting results: {collect_reults_cmd}')

        # Extract the directory path
        results_dir = os.path.dirname(results_file)
        # Create the directory if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        ret_code = BayesianExperiment.run_command(collect_reults_cmd, results_dir)
        if ret_code != 0:
            raise RuntimeError(f'Error: collecting experiment results failed with error code: {ret_code}')
        if os.path.exists(results_file):
            results_df = Utils.load_dataframe(results_file)
        else:
            results_df = pd.DataFrame()

        return results_df

    def convert_to_gray(binary):
        if isinstance(binary, str):
            binary = bitarray(binary)
        elif str(binary).isnumeric():
            binary = bitarray(bin(binary)[2:])
        gray = bitarray(0)
        gray.append(binary[0])
        for i in range(1, len(binary)):
            gray.append(binary[i] ^ binary[i-1])
        return gray

    def set_bits(bitarray_obj, bits_val):
        bits_to_set = bin(bits_val)[2:]
        bitarray_bits = bitarray_obj.to01()
        max_len = max(len(bits_to_set), len(bitarray_bits))
        bits_to_set = bits_to_set.zfill(max_len)
        bitarray_bits = bitarray_bits.zfill(max_len)
        new_bitarray = bitarray(bits_to_set) | bitarray(bitarray_bits)
        return new_bitarray

    def convert_from_gray(gray):
        if isinstance(gray, str):
            gray = bitarray(gray)
        elif str(gray).isnumeric():
            gray = bitarray(bin(gray)[2:])
        binary = bitarray(0)
        binary.append(gray[0])
        for i in range(1, len(gray)):
            binary.append(binary[i-1] ^ gray[i])
        return binary

    def convert_mem_layout_to_gray(self, mem_layout_hugepages):
        mem_layout_bin = bitarray(self.gray_layout_bit_vector_length)
        mem_layout_bin.setall(0)
        # createa one long bit-vector that represents the memory layout
        for p in mem_layout_hugepages:
            # mem_layout_bin = set_bits(mem_layout_bin, p)
            aggregated_p = int(p // self.hugepages_in_compressed_hugepage)
            mem_layout_bin[aggregated_p] = 1
        # reverse the string to make it readable as binary string
        mem_layout_bin.reverse()
        # convert to gray-code
        gray_mem_layout = BayesianExperiment.convert_to_gray(mem_layout_bin)
        gray_mem_layout.reverse()
        return gray_mem_layout

    def convert_dimensions_to_mem_layout_bin(self, mem_layout_dimensions):
        gray_mem_layout = bitarray(0)
        for i in range(len(mem_layout_dimensions)):
            gray_word = bin(mem_layout_dimensions[i])[2:]
            padding_size = self.dimension_size_in_bits
            if i == (len(mem_layout_dimensions) - 1):
                padding_size = self.last_dimension_size_in_bits
            padded_word = gray_word.zfill(padding_size)
            gray_mem_layout.extend(padded_word)
        gray_mem_layout.reverse()
        mem_layout = BayesianExperiment.convert_from_gray(gray_mem_layout)
        mem_layout.reverse()
        return mem_layout

    def decompress_memory_layout(self, mem_layout_dimensions):
        hugepages_bit_vector = self.convert_dimensions_to_mem_layout_bin(mem_layout_dimensions)
        mem_layout_hugepages = []
        for i in range(len(hugepages_bit_vector)):
            if hugepages_bit_vector[i] == 1:
                for k in range(self.hugepages_in_compressed_hugepage):
                    hugepage_idx = i * self.hugepages_in_compressed_hugepage + k
                    mem_layout_hugepages.append(hugepage_idx)
        return mem_layout_hugepages

    def compress_memory_layout(self, mem_layout_hugepages):
        gray_mem_layout = self.convert_mem_layout_to_gray(mem_layout_hugepages)

        compressed_mem_layout = [0] * self.num_dimensions
        for i in range(self.num_dimensions):
            dimension_start_idx = i*self.dimension_size_in_bits
            dimension_end_idx = dimension_start_idx + self.dimension_size_in_bits
            if dimension_start_idx >= len(gray_mem_layout):
                print('WARNING: memory layout size in gray code is smaller than in normal binary code')
                sys.exit(1)
                break
            if i == (self.num_dimensions - 1):
                dimension_end_idx = dimension_start_idx + self.last_dimension_size_in_bits
            gray_i = gray_mem_layout[dimension_start_idx:dimension_end_idx]
            gray_i.reverse()
            gray_i_number = int(gray_i.to01(), 2)
            compressed_mem_layout[i] = gray_i_number

        return compressed_mem_layout

    def predictTlbMisses(self, mem_layout):
        assert self.pebs_df is not None
        expected_tlb_coverage = self.pebs_df.query(f'PAGE_NUMBER in {mem_layout}')['NUM_ACCESSES'].sum()
        expected_tlb_misses = self.total_misses - expected_tlb_coverage
        print(f'[DEBUG]: mem_layout of size {len(mem_layout)} has an expected-tlb-coverage={expected_tlb_coverage} and expected-tlb-misses={expected_tlb_misses}')
        return expected_tlb_misses

    def generate_layout_from_pebs(self, pebs_coverage):
        assert self.pebs_df is not None

        df = self.pebs_df.sort_values('TLB_COVERAGE', ascending=False)

        mem_layout = []
        total_weight = 0
        for index, row in df.iterrows():
            page = row['PAGE_NUMBER']
            weight = row['TLB_COVERAGE']
            if (total_weight + weight) < (pebs_coverage + 0.5):
                mem_layout.append(page)
                total_weight += weight
            if total_weight >= pebs_coverage:
                break
        return mem_layout

    def get_layout_results(self, layout_name):
        results_df = BayesianExperiment.collect_results(self.collect_reults_cmd, self.results_file)
        layout_results = results_df[results_df['layout'] == layout_name]
        tlb_misses = layout_results['stlb_misses'].iloc[0]
        runtime = layout_results['cpu_cycles'].iloc[0]
        return tlb_misses, runtime

    def fill_buckets(self, buckets_weights, start_from_tail=False, fill_min_buckets_first=True):
        assert self.pebs_df is not None

        group_size = len(buckets_weights)
        group = [ [] for _ in range(group_size) ]
        df = self.pebs_df.sort_values('TLB_COVERAGE', ascending=start_from_tail)

        threshold = 2
        i = 0
        for index, row in df.iterrows():
            page = row['PAGE_NUMBER']
            weight = row['TLB_COVERAGE']
            selected_weight = None
            selected_index = None
            completed_buckets = 0
            # count completed buckets and find bucket with minimal remaining
            # space to fill, i.e., we prefer to place current page in the
            # bicket that has the lowest remaining weight/space
            for i in range(group_size):
                if buckets_weights[i] <= 0:
                    completed_buckets += 1
                elif buckets_weights[i] >= weight - threshold:
                    if selected_index is None:
                        selected_index = i
                        selected_weight = buckets_weights[i]
                    elif fill_min_buckets_first and buckets_weights[i] < selected_weight:
                        selected_index = i
                        selected_weight = buckets_weights[i]
                    elif not fill_min_buckets_first and buckets_weights[i] > selected_weight:
                        selected_index = i
                        selected_weight = buckets_weights[i]
            if completed_buckets == group_size:
                break
            # if there is a bucket that has a capacity of current page, add it
            if selected_index is not None:
                group[selected_index].append(page)
                buckets_weights[selected_index] -= weight
        return group

    def moselect_initial_samples(self):
        # desired weights for each group layout
        buckets_weights = [56, 28, 14]
        group = self.fill_buckets(buckets_weights)
        mem_layouts = []
        # create eight layouts as all subgroups of these three group layouts
        for subset_size in range(len(group)+1):
            for subset in itertools.combinations(group, subset_size):
                subset_pages = list(itertools.chain(*subset))
                mem_layouts.append(subset_pages)
        return mem_layouts

    def chebyshev_tlb_misses_initial_samples(self, num_samples, min_misses, max_misses):
        chebyshev_dist = (chebgauss(num_samples)[0] + np.ones(num_samples)) * 0.5
        range_misses = max_misses - min_misses
        samples_misses = chebyshev_dist * range_misses + min_misses
        samples_misses = samples_misses.astype(np.uint64)

        mem_layouts = []
        for w in samples_misses:
            layout = self.generate_layout_from_pebs(w)
            mem_layouts.append(layout)

        return mem_layouts

    def chebyshev_initial_samples(self, num_samples):
        '''
        Generate initial samples for Bayesian optimization using
        Chebyshev distribution with discrete integer dimensions.
        Use roots_chebyt to obtain the Chebyshev nodes,
        scales the values to match the desired range,
        and rounds them to the nearest integer to align
        with the Integer dimension.
        '''
        chebyshev_dist = (chebgauss(num_samples)[0] + np.ones(num_samples)) * 0.5
        chebyshev_dist = chebyshev_dist.reshape((num_samples, 1))
        dimensions_space = np.full((1, self.num_dimensions), fill_value=self.dimension_max_val)
        dimensions_space[0,-1] = self.last_dimension_max_val

        samples = chebyshev_dist * dimensions_space
        samples = samples.astype(np.uint64)

        decompressed_samples = [self.decompress_memory_layout(s) for s in samples]
        return decompressed_samples

    def generate_random_layout(self):
        mem_layout = []
        random_mem_layout = np.random.randint(2, size=self.num_default_hugepages)
        for i in range(len(random_mem_layout)):
            if random_mem_layout[i] == 1:
                mem_layout.append(i)
        return mem_layout

    def random_initial_samples(self, num_initial_layouts):
        mem_layouts = []
        for i in range(num_initial_layouts):
            random_mem_layout = self.generate_random_layout()
            mem_layouts.append(random_mem_layout)
        return mem_layouts

    def base_mem_layouts(self):
        base_pages_layout = []
        hugepages_layout = [i for i in range(self.num_default_hugepages)]
        mem_layouts = [base_pages_layout, hugepages_layout]
        return mem_layouts

    def get_previous_run_samples(self):
        X0 = []
        Y0 = []
        res_df = BayesianExperiment.collect_results(self.collect_reults_cmd, self.results_file)
        if res_df.empty:
            return X0, Y0
        for index, row in res_df.iterrows():
            layout_name = row['layout']
            mem_layout_pages = Utils.load_layout_hugepages(layout_name, self.exp_root_dir)
            tlb_misses = row['stlb_misses']
            compressed_mem_layout = self.compress_memory_layout(mem_layout_pages)
            X0.append(compressed_mem_layout)
            Y0.append(tlb_misses)
            self.last_layout_num += 1
        return X0, Y0

    def generate_initial_samples(self, num_initial_points, type):
        X0, Y0 = self.get_previous_run_samples()
        num_prev_samples = len(X0) if X0 else 0

        if type == 'base':
            mem_layouts = self.base_mem_layouts()
        elif type == 'random':
            mem_layouts = []
        elif type == 'our_random':
            mem_layouts = self.random_initial_samples(num_initial_points)
        elif type == 'chebyshev':
            mem_layouts = self.chebyshev_initial_samples(num_initial_points)
        elif type == 'chebyshev_misses':
            X0, Y0 = self.generate_initial_samples(2, 'base')
            mem_layouts = self.chebyshev_tlb_misses_initial_samples(8, Y0[1], Y0[0])
        elif type == 'moselect':
            mem_layouts = self.moselect_initial_samples()
        else:
            raise ValueError(f'Invalid initialization type to generate initial samples: {type}')
        for i, mem_layout in enumerate(mem_layouts):
            if i < num_prev_samples:
                self.last_layout_num += 1
                continue
            print(f'** Producing initial sample #{i} using a memory layout with {len(mem_layout)*self.hugepages_in_compressed_hugepage} (x2MB) hugepages')
            compressed_mem_layout = self.compress_memory_layout(mem_layout)
            X0.append(compressed_mem_layout)
            self.last_layout_num += 1
            layout_name = f'layout{self.last_layout_num}'
            tlb_misses = self.run_workload(compressed_mem_layout, layout_name)
            Y0.append(tlb_misses) # evaluate the objective function for each sample
        return X0, Y0

    def run_workload(self, compressed_mem_layout, layout_name):
        mem_layout = self.decompress_memory_layout(compressed_mem_layout)
        Utils.write_layout(layout_name, mem_layout, self.exp_root_dir, self.brk_footprint, self.mmap_footprint)

        print('--------------------------------------')
        print(f'** Running {layout_name} with {len(mem_layout)} hugepages')
        out_dir = f'{self.exp_root_dir}/{layout_name}'
        run_bayesian_cmd = f'{self.run_experiment_cmd} {layout_name}'
        ret_code = BayesianExperiment.run_command(run_bayesian_cmd, out_dir)
        if ret_code != 0:
            raise RuntimeError(f'Error: running {layout_name} failed with error code: {ret_code}')
        tlb_misses, runtime = self.get_layout_results(layout_name)
        print(f'\tResults: runtime={runtime/1e9:.2f} Billion cycles , stlb-misses={tlb_misses/1e9:.2f} Billions')
        print('--------------------------------------')
        return tlb_misses

    # Define the objective function using named arguments and the use_named_args decorator
    # @use_named_args(self.dimensions)
    def objective_function(self, mem_layout):
        # mem_layout = [params[f'mem_region_{i}'] for i in range(self.num_dimensions)]
        self.last_layout_num += 1
        layout_name = f'layout{self.last_layout_num}'
        return self.run_workload(mem_layout, layout_name)

    def run(self, initial_points=10, initialization_type='base'):
        # Define the initial data samples (X and Y pairs) for Bayesian optimization
        X0, Y0 = self.generate_initial_samples(initial_points, initialization_type)

        num_layouts = max(0, (self.num_layouts - len(X0)))
        if num_layouts == 0:
            print('================================================')
            print(f'No more layouts to run for the experiment:\n{self.exp_root_dir}')
            print('================================================')
            return

        # Perform Bayesian optimization with the initial data samples
        result = gp_minimize(self.objective_function,  # the objective function to minimize
                            dimensions=self.dimensions,  # the search space
                            acq_func='EI',  # the acquisition function
                            n_calls=num_layouts,  # the number of evaluations of f including at x0
                            x0=X0,  # the initial data samples
                            y0=Y0)  # the initial data sample evaluations

        print('================================================')
        print(f'Finished running Bayesian Optimization process for:\n{self.exp_root_dir}')
        print("result:", result)
        print('================================================')
        # print("Best TLB misses:", result.fun)
        # compressed_best_layout = [int(x) for x in result.x]
        # print("Best memory layout (compressed):", compressed_best_layout)
        # decompressed_best_layout = self.decompress_memory_layout(compressed_best_layout)
        # print(f"Best memory layout ({len(decompressed_best_layout)} items):")
        # if len(decompressed_best_layout) <= 20:
        #     print(decompressed_best_layout)
        # else:
        #     print(decompressed_best_layout[:10], '...', decompressed_best_layout[-10:])

import argparse
def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--memory_footprint', default='memory_footprint.txt')
    parser.add_argument('-p', '--pebs_mem_bins', default=None)
    parser.add_argument('-e', '--exp_root_dir', required=True)
    parser.add_argument('-r', '--results_file', required=True)
    parser.add_argument('-c', '--collect_reults_cmd', required=True)
    parser.add_argument('-x', '--run_experiment_cmd', required=True)
    parser.add_argument('-n', '--num_layouts', required=True, type=int)
    parser.add_argument('-i', '--initialization_method', choices=['base', 'random', 'chebyshev', 'chebyshev_misses', 'moselect'], default='base')
    parser.add_argument('-d', '--debug', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parseArguments()

    # profiler = cProfile.Profile()
    # profiler.enable()
    exp = BayesianExperiment(args.memory_footprint, args.pebs_mem_bins,
                             args.collect_reults_cmd, args.results_file,
                             args.run_experiment_cmd, args.exp_root_dir,
                             args.num_layouts)
    exp.run(initialization_type=args.initialization_method)
    # profiler.disable()
    # profiler.dump_stats('profile_results.prof')
    # profiler.print_stats()
