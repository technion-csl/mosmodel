#!/usr/bin/env python3

import sys
import os
import pandas as pd
import numpy as np
import random
import math
from Utils.utils import *
from Utils.ConfigurationFile import *

kb = 1024
mb = 1024*kb
gb = 1024*mb

def round_up(x, base):
    return int(base * math.ceil(x/base))

def round_down(x, base):
    return (int(x / base) * base)

class LayoutsGenerator:
    def __init__(self, memory_footprint, num_layouts, use_1gb_pages):
        self._memory_footprint = memory_footprint
        self._num_layouts = num_layouts
        self._use_1gb_pages = use_1gb_pages
        self._layouts = []

        footprint_df = pd.read_csv(memory_footprint)
        mmap_footprint = footprint_df['anon-mmap-max'][0]
        self._mmap_footprint = round_up(mmap_footprint, 2*mb)
        brk_footprint = footprint_df['brk-max'][0]
        self._brk_footprint = round_up(brk_footprint, 2*mb)
        self._mmap_2mb_footprint = round_up(self._mmap_footprint, 2*mb)
        self._brk_2mb_footprint = round_up(self._brk_footprint, 2*mb)
        self._mmap_1gb_footprint = round_up(self._mmap_footprint, gb)
        self._brk_1gb_footprint = round_up(self._brk_footprint, gb)
        self._mmap_pool_size = self._mmap_footprint
        self._brk_pool_size = self._brk_footprint


    def addSingleWindowLayout(self,
            brk_start_2m=0, brk_end_2m=0,
            mmap_start_2m=0, mmap_end_2m=0,
            brk_start_1g=0, brk_end_1g=0,
            mmap_start_1g=0, mmap_end_1g=0,
            add_to_front=False):
        brk_footprint = max(brk_end_2m, brk_end_1g, self._brk_footprint)
        mmap_footprint = max(mmap_end_2m, mmap_end_1g, self._mmap_footprint)
        self._mmap_pool_size = max(mmap_footprint, self._mmap_pool_size)
        self._brk_pool_size = max(brk_footprint, self._brk_pool_size)
        configuration = Configuration()
        if brk_start_2m != brk_end_2m:
            configuration.addWindow(
                    type=configuration.TYPE_BRK,
                    page_size=Configuration.HUGE_2MB_PAGE_SIZE,
                    start_offset=brk_start_2m,
                    end_offset=brk_end_2m)
        if brk_start_1g != brk_end_1g:
            configuration.addWindow(
                    type=configuration.TYPE_BRK,
                    page_size=Configuration.HUGE_1GB_PAGE_SIZE,
                    start_offset=brk_start_1g,
                    end_offset=brk_end_1g)
        if mmap_start_2m != mmap_end_2m:
            configuration.addWindow(
                    type=configuration.TYPE_MMAP,
                    page_size=Configuration.HUGE_2MB_PAGE_SIZE,
                    start_offset=mmap_start_2m,
                    end_offset=mmap_end_2m)
        if mmap_start_1g != mmap_end_1g:
            configuration.addWindow(
                    type=configuration.TYPE_MMAP,
                    page_size=Configuration.HUGE_1GB_PAGE_SIZE,
                    start_offset=mmap_start_1g,
                    end_offset=mmap_end_1g)

        if add_to_front:
            self._layouts.insert(0, configuration)
        else:
            self._layouts.append(configuration)

    def buildGrowingWindowLayouts(self, max_1gb_hugepages):
        step = int(math.floor(self._brk_footprint / (self._num_layouts-1)))
        step = round_up(step, 4*kb)
        for i in range(self._num_layouts-1):
            step_size = step * i
            region_2m_size = round_down(step_size, 2*mb)
            end_1g = 0
            if self._use_1gb_pages:
                end_1g = int(region_2m_size / gb) * gb
                end_1g = min(end_1g, max_1gb_hugepages)
            region_2m_size -= end_1g
            end_2m = end_1g + region_2m_size

            layout_start_2m = 0
            layout_end_2m = 0
            if end_1g != end_2m:
                layout_start_2m = end_1g
                layout_end_2m = end_2m
            # reverse layouts list order to enforce running layouts with more
            # huge pages first in growing_window to prevent huge-pages-reservation failures
            # in case of memory fragmentation (and then huge-pages reservation fails)
            self.addSingleWindowLayout(
                    brk_start_2m=layout_start_2m, brk_end_2m=layout_end_2m,
                    mmap_start_2m=0, mmap_end_2m=self._mmap_2mb_footprint,
                    brk_start_1g=0, brk_end_1g=end_1g,
                    mmap_start_1g=0, mmap_end_1g=0,
                    add_to_front=True)
        #add all 1gb pages
        brk_rounded_footprint = max(min(max_1gb_hugepages, self._brk_1gb_footprint), self._brk_2mb_footprint)
        brk_end_1gb = min(max_1gb_hugepages, round_down(brk_rounded_footprint, gb))

        if self._use_1gb_pages:
            mmap_pool_size = self._mmap_2mb_footprint
            brk_start_2mb = brk_end_1gb
            brk_end_2mb = brk_rounded_footprint
        else:
            mmap_pool_size = self._mmap_2mb_footprint
            brk_start_2mb = 0
            brk_end_2mb = self._brk_2mb_footprint
            brk_end_1gb = 0

        self.addSingleWindowLayout(
                brk_start_2m=brk_start_2mb, brk_end_2m=brk_end_2mb,
                mmap_start_2m=0, mmap_end_2m=self._mmap_2mb_footprint,
                brk_start_1g=0, brk_end_1g=brk_end_1gb,
                mmap_start_1g=0, mmap_end_1g=0,
                add_to_front=True)

    def buildRandomWindowLayouts(self, seed=0, window_min_size_ratio=0):
        start_offset = 0
        end_offset = self._brk_footprint
        window_min_size = max(2*mb, round_up(window_min_size_ratio * self._brk_footprint, 2*mb))

        random.seed(seed)
        for i in range(self._num_layouts):
            random_start_offset = random.randrange(start_offset,
                    end_offset - window_min_size, (4*kb))
            random_end_offset = random.randrange(random_start_offset + window_min_size,
                    end_offset, 2*mb)

            self.addSingleWindowLayout(brk_start_2m=random_start_offset, brk_end_2m=random_end_offset)

    def buildSlidingWindowLayouts(self, hot_region_start, hot_region_length):
        standard_page_size = 4*kb
        window_page_size = 2*mb
        if self._use_1gb_pages:
            window_page_size = 1*gb

        window_start = round_down(hot_region_start, standard_page_size)
        raw_window_length = round_up(hot_region_length, standard_page_size)
        window_length = round_up(raw_window_length, window_page_size)

        window_end = window_start + window_length

        brk_footprint = max(self._brk_footprint, window_end)

        # check where to move the window: to left or right (to the direction with enough space)
        if window_start > window_length:
            start_offset = window_start - window_length
        elif (window_length + window_end) <= brk_footprint:
            start_offset = window_start
        else:
            start_offset = window_start
            brk_footprint = window_end + window_length

        window_overcommit_max_size = 5 * standard_page_size
        if (start_offset + 2*window_length) > brk_footprint and \
            (start_offset + 2*window_length) <= (brk_footprint + window_overcommit_max_size):
                brk_footprint = (start_offset + 2*window_length)

        if (start_offset + 2*window_length) > brk_footprint:
            sys.exit(str.format('window <{0} : {1}> of the benchmark exceeds the benchmark memory footprint <{2}>',
                start_offset, (start_offset + 2*window_length), brk_footprint))

        step_size = math.floor(raw_window_length / (self._num_layouts-1))
        step_size = round_up(step_size, 4*kb)

        for i in range(0, self._num_layouts):
            end_offset = (start_offset + window_length)
            page_size = 2*mb
            if self._use_1gb_pages:
                end_offset = round_up(end_offset, gb)
                self.addSingleWindowLayout(brk_start_1g=start_offset, brk_end_1g=end_offset)
            else:
                self.addSingleWindowLayout(brk_start_2m=start_offset, brk_end_2m=end_offset)
            start_offset += step_size

    def exportLayouts(self, output):
        i=1
        for l in self._layouts:
            l.setPoolsSize(
                    brk_size=self._brk_pool_size,
                    file_size=1*gb,
                    mmap_size=self._mmap_pool_size)
            l.exportToCSV(output, 'layout' + str(i))
            i += 1


