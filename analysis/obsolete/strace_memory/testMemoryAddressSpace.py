#! /usr/bin/env python3

import unittest
from memory_address_space import MemoryAddressSpace

class TestMemoryAddressSpace(unittest.TestCase):
    def testSimpleInputFile(self):
        mas = MemoryAddressSpace()
        input_file1 = 'test_strace1.out'
        with open(input_file1, 'r') as fid:
            mas.followStraceFile(fid)
        
        self.assertEqual(mas.max_brk_pool_size, 135168)
        self.assertEqual(mas.max_anon_pool_size, 27040)
        self.assertEqual(mas.max_file_pool_size, 6977344)
        
unittest.main()
        