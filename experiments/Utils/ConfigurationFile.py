import pandas as pd
import os


class Configuration:
    TYPE = "type"
    PAGE_SIZE = "pageSize"
    START_OFFSET = "startOffset"
    END_OFFSET = "endOffset"
    TYPE_FILE = "file"
    TYPE_MMAP = "mmap"
    TYPE_BRK = "brk"
    POOL_SIZE_FLAG = -1
    LAYOUT_DIRECTORY = "layouts"
    HUGE_2MB_PAGE_SIZE = 2097152
    HUGE_1GB_PAGE_SIZE = 1073741824

    def __init__(self):
        self.config = []

    def addWindow(self, type, page_size, start_offset, end_offset):
        row = {
            self.END_OFFSET: end_offset,
            self.START_OFFSET: start_offset,
            self.PAGE_SIZE: page_size,
            self.TYPE: type

        }
        self.config.append(row)

    def setPoolsSize(self, brk_size, file_size, mmap_size):
        if brk_size is not None:
            self.addWindow(
                type=Configuration.TYPE_BRK, page_size=self.POOL_SIZE_FLAG, start_offset=0, end_offset=brk_size)
        if file_size is not None:
            self.addWindow(type=self.TYPE_FILE,
                           page_size=self.POOL_SIZE_FLAG, start_offset=0, end_offset=file_size)
        if mmap_size is not None:
            self.addWindow(type=self.TYPE_MMAP,
                           page_size=self.POOL_SIZE_FLAG, start_offset=0, end_offset=mmap_size)

    def mergeAdjacentWindows(df, pool_type, page_size):
        df = df[df[Configuration.TYPE] == pool_type]
        df = df[df[Configuration.PAGE_SIZE] == page_size]
        df = df.sort_values(Configuration.START_OFFSET)
        start_offset = -1
        windows = pd.DataFrame(columns=df.columns)
        for index, row in df.iterrows():
            curr_start_offset = row[Configuration.START_OFFSET]
            curr_end_offset = row[Configuration.END_OFFSET]
            if start_offset == -1:
                start_offset = curr_start_offset
                end_offset = curr_end_offset
            elif end_offset == row[Configuration.START_OFFSET]:
                end_offset = row[Configuration.END_OFFSET]
            else:
                row[Configuration.START_OFFSET] = start_offset
                row[Configuration.END_OFFSET] = end_offset
                start_offset = curr_start_offset
                end_offset = curr_end_offset
                windows = pd.concat([windows, row.to_frame().T], ignore_index=True)
        if start_offset != -1:
            row[Configuration.START_OFFSET] = start_offset
            row[Configuration.END_OFFSET] = end_offset
            windows = pd.concat([windows, row.to_frame().T], ignore_index=True)
        return windows


    def exportToCSV(self, abs_path, layout_name):
        df = pd.DataFrame(self.config, index=None)
        # rearrange columns
        df = df[[Configuration.TYPE, Configuration.PAGE_SIZE,
                 Configuration.START_OFFSET, Configuration.END_OFFSET]]
        windows = df[df[Configuration.PAGE_SIZE] == Configuration.POOL_SIZE_FLAG]
        brk_2mb_window = Configuration.mergeAdjacentWindows(df, Configuration.TYPE_BRK, Configuration.HUGE_2MB_PAGE_SIZE)
        windows = pd.concat([windows, brk_2mb_window])
        brk_1gb_window = Configuration.mergeAdjacentWindows(df, Configuration.TYPE_BRK, Configuration.HUGE_1GB_PAGE_SIZE)
        windows = pd.concat([windows, brk_1gb_window])
        path = os.path.join(abs_path, Configuration.LAYOUT_DIRECTORY)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        export_file_name = os.path.join(path, layout_name + ".csv")
        windows.to_csv(export_file_name, index=None)


