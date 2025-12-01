#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(sys.argv[0])+"/..")
from layouts_generator import LayoutsGenerator

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--memory_footprint', default='memory_footprint.txt')
parser.add_argument('-n', '--num_layouts', type=int, default=9)
parser.add_argument('-w', '--window_min_size_ratio', type=float, default=0)
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('-o', '--output', required=True)
args = parser.parse_args()

generator = LayoutsGenerator(args.memory_footprint, args.num_layouts, False)
generator.buildRandomWindowLayouts(args.seed, args.window_min_size_ratio)
generator.exportLayouts(args.output)


