/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2020  CP2K developers group                         *
 *****************************************************************************/

#include <cassert>
#include <cstdio>
#include <iostream>
#include "grid_base_ref_replay.h"

#include "rt_graph.hpp"

rt_graph::Timer timer;

int main(int argc, char *argv[]){
    if (argc != 2) {
        printf("Usage: grid_base_ref_miniapp.x <task-file>\n");
        return 1;
    }
    const int cycles = 10000;  // For better statistics the task is collocated many times.
    timer.start("grid_collocate_replay");
    const double max_diff = grid_collocate_replay(argv[1], cycles);
    timer.stop("grid_collocate_replay");
    assert(max_diff < 1e-18 * cycles);

    // process timings
    const auto result = timer.process();
    // print default statistics
    std::cout << "Default statistic:" << std::endl;
    std::cout << result.print();

    return 0;
}

//EOF
