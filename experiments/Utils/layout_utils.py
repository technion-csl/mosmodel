#!/usr/bin/env python3
# import cProfile
import pandas as pd
from Utils.utils import Utils
from Utils.ConfigurationFile import Configuration

class LayoutUtils:
    def __init__(self) -> None:
        pass
        
    def load_layout_hugepages(layout_name, exp_dir):
        hugepage_size = 1 << 21
        base_page_size = 1 << 12
        layout_file = str.format('{exp_root}/layouts/{layout_name}.csv',
                exp_root=exp_dir,
                layout_name=layout_name)
        df = pd.read_csv(layout_file)
        df = df[df['type'] == 'brk']
        df = df[df['pageSize'] == hugepage_size]
        pages = []
        offset = 0
        for index, row in df.iterrows():
            start_page = int(row['startOffset'] / hugepage_size)
            end_page = int(row['endOffset'] / hugepage_size)
            offset = int(row['startOffset'] % hugepage_size)
            pages += list(range(start_page, end_page))
        start_offset = offset / base_page_size
        return pages
    
    def write_layout(layout, pages, output, brk_footprint, mmap_footprint, sliding_index=0):
        hugepage_size= 1 << 21
        base_page_size = 1 << 12
        hugepages_start_offset = sliding_index * base_page_size
        brk_pool_size = Utils.round_up(brk_footprint, hugepage_size) + hugepages_start_offset
        configuration = Configuration()
        configuration.setPoolsSize(
                brk_size=brk_pool_size,
                file_size=1*Utils.GB,
                mmap_size=mmap_footprint)
        for p in pages:
            configuration.addWindow(
                    type=configuration.TYPE_BRK,
                    page_size=hugepage_size,
                    start_offset=(p * hugepage_size) + hugepages_start_offset,
                    end_offset=((p+1) * hugepage_size) + hugepages_start_offset)
        configuration.exportToCSV(output, layout)
