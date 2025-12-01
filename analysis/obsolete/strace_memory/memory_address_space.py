#! /usr/bin/env python3

class MemoryAddressSpace:
    def __init__(self):
        self.max_brk_pool_size = 0
        self.max_anon_pool_size = 0
        self.max_file_pool_size = 0
        
        self._current_heap_size = 0
        self._current_anon_pool_size = 0
        self._current_file_pool_size = 0
        
        self._heap_bottom = None
        self._heap_top = None
        self._anon_mappings = []
        self._file_mappings = []
        
    def _parseMmapCall(self, mmap_args, mmap_return_value):
        mmap_return_value = int(mmap_return_value, 0)
        assert(mmap_return_value != -1)
        
        flags = mmap_args[3]
        
        addr = mmap_args[0]
        if addr == 'NULL':
            addr = mmap_return_value
        else:
            assert('MAP_FIXED' in flags)
            addr = int(addr, 0) # the second argument is 0 to guess the 0x base
        assert(addr % 2**12 == 0)
        
        length = int(mmap_args[1])
        assert(length >= 0)
        region = (addr, addr+length)
        
        fd = int(mmap_args[4])
        if (fd == -1):
            assert('MAP_ANONYMOUS' in flags)
            self._anon_mappings.append(region)
            self._current_anon_pool_size += length
            if self._current_anon_pool_size > self.max_anon_pool_size:
                self.max_anon_pool_size = self._current_anon_pool_size
        else:
            assert(fd >= 0)
            assert('MAP_ANONYMOUS' not in flags)
            self._file_mappings.append(region)
            self._current_file_pool_size += length
            if self._current_file_pool_size > self.max_file_pool_size:
                self.max_file_pool_size = self._current_file_pool_size
        
    def _parseMunmapCall(self, munmap_args, munmap_return_value):
        munmap_return_value = int(munmap_return_value, 0)
        assert(munmap_return_value != -1)
        
        addr = munmap_args[0]
        assert(addr != 'NULL')
        addr = int(addr, 0) # the second argument is 0 to guess the 0x base
        assert(addr % 2**12 == 0)
        
        length = int(munmap_args[1])
        assert(length >= 0)
        region = (addr, addr+length)
        
        if region in self._anon_mappings:
            self._anon_mappings.remove(region)
            self._current_anon_pool_size -= length
        elif region in self._file_mappings:
            self._file_mappings.remove(region)
            self._current_file_pool_size -= length
        else:
            #print(self._anon_mappings)
            raise ValueError('A call to munmap with non-existent region: ' + \
                             ''.join(munmap_args))
        
    def _parseBrkCall(self, brk_args, brk_return_value):
        addr = brk_args[0]
        brk_return_value = int(brk_return_value, 0)
        
        if addr == 'NULL':
            addr = 0
            self._heap_bottom = brk_return_value
            self._heap_top = self._heap_bottom
        else:
            # assert that brk() didn't fail. From the brk() man page:
            # "On failure, the system call returns the current break."
            assert(brk_return_value != self._heap_top)
            self._heap_top = brk_return_value
            
        heap_size = self._heap_top - self._heap_bottom
        if heap_size > self._current_heap_size:
            self.max_brk_pool_size = heap_size
        
    def followStraceFile(self, input_fid):
        for line in input_fid:
            if line.startswith('--- SIG') or line.startswith('+++ exited'):
                continue
            
            syscall = line.split('(')[0]
            syscall_args = line.split('(')[1].split(')')[0].split(',')
            syscall_return_value = line.split('=')[1]
            
            if syscall == 'mmap':
                self._parseMmapCall(syscall_args, syscall_return_value)
            elif syscall == 'munmap':
                self._parseMunmapCall(syscall_args, syscall_return_value)
            elif syscall == 'brk':
                self._parseBrkCall(syscall_args, syscall_return_value)
            else:   # other memory syscalls, e.g., mprotect
                continue
