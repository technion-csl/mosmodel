/* -*- mode: C; tab-width: 2; indent-tabs-mode: nil; -*- */

/*
 * This code has been contributed by the DARPA HPCS program.  Contact
 * David Koester <dkoester@mitre.org> or Bob Lucas <rflucas@isi.edu>
 * if you have questions.
 *
 * GUPS (Giga UPdates per Second) is a measurement that profiles the memory
 * architecture of a system and is a measure of performance similar to MFLOPS.
 * The HPCS HPCchallenge RandomAccess benchmark is intended to exercise the
 * GUPS capability of a system, much like the LINPACK benchmark is intended to
 * exercise the MFLOPS capability of a computer.  In each case, we would
 * expect these benchmarks to achieve close to the "peak" capability of the
 * memory system. The extent of the similarities between RandomAccess and
 * LINPACK are limited to both benchmarks attempting to calculate a peak system
 * capability.
 *
 * GUPS is calculated by identifying the number of memory locations that can be
 * randomly updated in one second, divided by 1 billion (1e9). The term "randomly"
 * means that there is little relationship between one address to be updated and
 * the next, except that they occur in the space of one half the total system
 * memory.  An update is a read-modify-write operation on a table of 64-bit words.
 * An address is generated, the value at that address read from memory, modified
 * by an integer operation (add, and, or, xor) with a literal value, and that
 * new value is written back to memory.
 *
 * We are interested in knowing the GUPS performance of both entire systems and
 * system subcomponents --- e.g., the GUPS rating of a distributed memory
 * multiprocessor the GUPS rating of an SMP node, and the GUPS rating of a
 * single processor.  While there is typically a scaling of FLOPS with processor
 * count, a similar phenomenon may not always occur for GUPS.
 *
 * For additional information on the GUPS metric, the HPCchallenge RandomAccess
 * Benchmark,and the rules to run RandomAccess or modify it to optimize
 * performance -- see http://icl.cs.utk.edu/hpcc/
 *
 */

/*
 * This file contains the computational core of the single cpu version
 * of GUPS.  The inner loop should easily be vectorized by compilers
 * with such support.
 *
 * This core is used by both the single_cpu and star_single_cpu tests.
 */

#include <cinttypes>
#include <cstddef>
#include <cassert>

#define POLY 0x0000000000000007UL
#define PERIOD 1317624576693539401L
/* utility routine to create a seed for the n-th parallel access */
uint64_t random_seed(uint64_t n) {
    int i, j;
    uint64_t m2[64];
    uint64_t temp, ran;

    while (n < 0) n += PERIOD;
    while (n > PERIOD) n -= PERIOD;
    if (n == 0) return 0x1;

    temp = 0x1;
    for (i=0; i<64; i++) {
        m2[i] = temp;
        temp = (temp << 1) ^ ((int64_t) temp < 0 ? POLY : 0);
        temp = (temp << 1) ^ ((int64_t) temp < 0 ? POLY : 0);
    }

    for (i=62; i>=0; i--)
        if ((n >> i) & 1)
            break;

    ran = 0x2;
    while (i > 0) {
        temp = 0;
        for (j=0; j<64; j++)
            if ((ran >> j) & 1)
                temp ^= m2[j];
        ran = temp;
        i -= 1;
        if ((n >> i) & 1)
            ran = (ran << 1) ^ ((int64_t) ran < 0 ? POLY : 0);
    }

    return ran;
}

int main(int argc, char* argv[]) {
    uint64_t log2_array_length = 27ul;
    uint64_t array_length = 1ul<<log2_array_length;
    /* Number of updates to table (suggested: 4x number of table entries) */
    uint64_t num_updates = array_length<<2ul; /* 4x the array size */
    const uint64_t num_parallel_accesses = 128;
    uint64_t* array = new uint64_t[array_length];
    assert(array != NULL);

    /* Initialize main table */
    for (uint64_t i = 0; i < array_length; i++) {
        array[i] = i;
    }

    /* Calculate the initial ran */
    uint64_t ran[128];
    for (uint64_t j = 0; j < num_parallel_accesses; j++) {
        ran[j] = random_seed((num_updates / num_parallel_accesses) * j);
    }

    /* Perform updates to main table */
    for (uint64_t i = 0; i < num_updates / num_parallel_accesses; i++) {
        for (uint64_t j = 0; j < num_parallel_accesses; j++) {
            ran[j] = (ran[j] << 1) ^ ((int64_t) ran[j] < 0 ? POLY : 0);
            array[ran[j] & (array_length-1)] ^= ran[j];
        }
    }

    delete[] array;

    return 0;
}

