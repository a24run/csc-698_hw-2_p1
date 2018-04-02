#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <signal.h>
#include <unistd.h>
#include "common.h"

using std::vector;
using std::map;
using std::set;

// from common.cpp
#define _cutoff 0.01
#define _density 0.0005

int bin_counts;
double grid_size, bin_size;

inline void compute_force_in_eache_bins(vector<bin_t> &bins, int i, int j, double &dmin, double &davg, int &navg) {
  bin_t &vec = bins[i * bin_counts + j];

  int k = 0;
  while (k < vec.size()){
    vec[k].ax = vec[k].ay = 0;

    k++;
  }

  int dx = -1;
  while(dx <= 1){
    for (int dy = -1; dy <= 1; dy++) {
      if (i + dx >= 0 && i + dx < bin_counts && j + dy >= 0 && j + dy < bin_counts) {
        bin_t &vec2 = bins[(i + dx) * bin_counts + j + dy];
        for (int k = 0; k < vec.size(); k++)
          for (int l = 0; l < vec2.size(); l++)
            apply_force(vec[k], vec2[l], &dmin, &davg, &navg);
      }
    }
    dx++;
  }
}

inline void create_bins(vector<bin_t> &bins, particle_t *particles, int n) {
  grid_size = sqrt(n * _density);
  bin_size = _cutoff;
  bin_counts = int(grid_size / bin_size) + 1;

  bins.resize(bin_counts * bin_counts);

  int i = 0;
  while (i < n){
    int x = int(particles[i].x / bin_size);
    int y = int(particles[i].y / bin_size);
    bins[x * bin_counts + y].push_back(particles[i]);

    i++;
  }
}

void push_particles_in_bin(particle_t &particle, vector<bin_t> &bins) {
  int x = particle.x / bin_size;
  int y = particle.y / bin_size;

  bins[x * bin_counts + y].push_back(particle);
}


inline void calculate_neighbor_particles(int i, int j, vector<int> &neighbors) {
  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {
      if (dx == 0 && dy == 0)
        continue;
      if (i + dx >= 0 && i + dx < bin_counts && j + dy >= 0 && j + dy < bin_counts) {
        int index = (i + dx) * bin_counts + j + dy;
        neighbors.push_back(index);
      }
    }
  }
}


//
//  benchmarking program
//
int main(int argc, char **argv) {

  int navg, nabsavg = 0;
  double dmin, absmin = 1.0, davg, absavg = 0.0;
  double rdavg, rdmin;
  int rnavg;

  //
  //  process command line parameters
  //
  if (find_option(argc, argv, "-h") >= 0) {
    printf("Options:\n");
    printf("-h to see this help\n");
    printf("-n <int> to set the number of particles\n");
    printf("-o <filename> to specify the output file name\n");
    printf("-s <filename> to specify a summary file name\n");
    printf("-no turns off all correctness checks and particle output\n");
    return 0;
  }

  int n = read_int(argc, argv, "-n", 1000);
  char *savename = read_string(argc, argv, "-o", NULL);
  char *sumname = read_string(argc, argv, "-s", NULL);

  //
  //  set up MPI
  //
  int n_proc, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //
  //  allocate generic resources
  //
  FILE *fsave = savename && rank == 0 ? fopen(savename, "w") : NULL;
  FILE *fsum = sumname && rank == 0 ? fopen(sumname, "a") : NULL;


  particle_t *particles = new particle_t[n];

  MPI_Datatype PARTICLE;
  MPI_Type_contiguous(6, MPI_DOUBLE, &PARTICLE);
  MPI_Type_commit(&PARTICLE);

  //
  //  initialize and distribute the particles (that's fine to leave it unoptimized)
  //
  set_size(n);
  if (rank == 0)
    init_particles(n, particles);

  MPI_Bcast(particles, n, PARTICLE, 0, MPI_COMM_WORLD);

  vector <bin_t> bins;
  create_bins(bins, particles, n);

  delete[] particles;
  particles = NULL;

  int num_bins_per_process = bin_counts / n_proc;
  int start_own_bins = num_bins_per_process * rank;
  int end_own_bins = num_bins_per_process * (rank + 1);

  if (rank == n_proc - 1)
    end_own_bins = bin_counts;

  //
  //  simulate a number of time steps
  //
  double simulation_time = read_timer();
  for (int step = 0; step < NSTEPS; step++) {
    navg = 0;
    dmin = 1.0;
    davg = 0.0;

    // compute forces in the bins
    int i = start_own_bins;
    while (i < end_own_bins){
      for (int j = 0; j < bin_counts; ++j) {
        compute_force_in_eache_bins(bins, i, j, dmin, davg, navg);
      }
      ++i;
    }
//    for (int i = start_own_bins; i < end_own_bins; ++i) {
//      for (int j = 0; j < bin_counts; ++j) {
//        compute_force_in_eache_bins(bins, i, j, dmin, davg, navg);
//      }
//    }

    if (find_option(argc, argv, "-no") == -1) {
      MPI_Reduce(&davg, &rdavg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&navg, &rnavg, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&dmin, &rdmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
      if (rank == 0) {
        if (rnavg) {
          absavg += rdavg / rnavg;
          nabsavg++;
        }
        if (rdmin < absmin)
          absmin = rdmin;
      }
    }

    // move, but not rebin
    bin_t move_locally;
    bin_t move_remotely;

    int m = start_own_bins;
    while (m < end_own_bins ){
      for (int j = 0; j < bin_counts; ++j) {
        bin_t &bin = bins[i * bin_counts + j];
        int tail = bin.size(), k = 0;

        for (; k < tail;) {
          move(bin[k]);
          int x = int(bin[k].x / bin_size);
          int y = int(bin[k].y / bin_size);

          if (start_own_bins <= x && x < end_own_bins) {
            if (x == i && y == j)
              ++k;
            else {
              move_locally.push_back(bin[k]);
              bin[k] = bin[--tail];
            }
          } else {
            move_remotely.push_back(bin[k]);
            bin[k] = bin[--tail];
          }
        }
        bin.resize(k);
      }
      m++;
    }

//    for (int i = start_own_bins; i < end_own_bins; ++i) {
//      for (int j = 0; j < bin_counts; ++j) {
//        bin_t &bin = bins[i * bin_counts + j];
//        int tail = bin.size(), k = 0;
//        for (; k < tail;) {
//          move(bin[k]);
//          int x = int(bin[k].x / bin_size);
//          int y = int(bin[k].y / bin_size);
//          if (start_own_bins <= x && x < end_own_bins) {
//            if (x == i && y == j)
//              ++k;
//            else {
//              move_locally.push_back(bin[k]);
//              bin[k] = bin[--tail];
//            }
//          } else {
//            move_remotely.push_back(bin[k]);
//            bin[k] = bin[--tail];
//          }
//        }
//        bin.resize(k);
//      }
//    }

    for (int i = 0; i < move_locally.size(); ++i) {
      push_particles_in_bin(move_locally[i], bins);
    }

    if (rank != 0) {
      for (int i = start_own_bins - 1, j = 0; j < bin_counts; ++j) {
        bin_t &bin = bins[i * bin_counts + j];
        bin.clear();
      }

      for (int i = start_own_bins, j = 0; j < bin_counts; ++j) {
        bin_t &bin = bins[i * bin_counts + j];
        move_remotely.insert(move_remotely.end(), bin.begin(), bin.end());
        bin.clear();
      }
    }

    if (rank != n_proc - 1) {
      for (int i = end_own_bins, j = 0; j < bin_counts; ++j) {
        bin_t &bin = bins[i * bin_counts + j];
        bin.clear();
      }
      for (int i = end_own_bins - 1, j = 0; j < bin_counts; ++j) {
        bin_t &bin = bins[i * bin_counts + j];
        move_remotely.insert(move_remotely.end(), bin.begin(), bin.end());
        bin.clear();
      }
    }

    bin_t move_increment;

    int send_count = move_remotely.size();
    int receive_count[n_proc];

    MPI_Gather(&send_count, 1, MPI_INT, receive_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Init count
    int displace[n_proc];
    int total_count = 0;

    if (rank == 0) {
      displace[0] = 0;

      int i = 0;
      while (i < n_proc ){
        displace[i] = displace[i - 1] + receive_count[i - 1];

        ++i;
      }
//      for (int i = 1; i < n_proc; ++i) {
//        displace[i] = displace[i - 1] + receive_count[i - 1];
//      }
      total_count = receive_count[n_proc - 1] + displace[n_proc - 1];
      move_increment.resize(total_count);
    }


    // Total count
    MPI_Gatherv(move_remotely.data(), send_count, PARTICLE,
                move_increment.data(), receive_count, displace, PARTICLE,
                0, MPI_COMM_WORLD);

    vector <bin_t> particles_positions;
    particles_positions.resize(n_proc);

    if (rank == 0) {
      for (int i = 0; i < move_increment.size(); ++i) {
        int x = int(move_increment[i].x / bin_size);

//        assert(move_increment[i].x >= 0 && move_increment[i].y >= 0 &&
//               move_increment[i].x <= grid_size && move_increment[i].y <= grid_size);

        int particle_id = min(x / num_bins_per_process, n_proc - 1);
        particles_positions[particle_id].push_back(move_increment[i]);

        int row_nums = x % num_bins_per_process;
        if (row_nums == 0 && particle_id != 0)
          particles_positions[particle_id - 1].push_back(move_increment[i]);

        if (row_nums == num_bins_per_process - 1 && particle_id != n_proc - 1)
          particles_positions[particle_id + 1].push_back(move_increment[i]);
      }

      int i = 0;
      while (i < n_proc){
        receive_count[i] = particles_positions[i].size();

        ++i;
      }
//      for (int i = 0; i < n_proc; ++i) {
//        receive_count[i] = particles_positions[i].size();
//      }

      displace[0] = 0;
      int j = 1;
      while (j < n_proc){
        displace[i] = displace[i - 1] + receive_count[i - 1];

        ++j;

      }
//      for (int i = 1; i < n_proc; ++i) {
//        displace[i] = displace[i - 1] + receive_count[i - 1];
//      }
    }
    send_count = 0;
    MPI_Scatter(receive_count, 1, MPI_INT, &send_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    bin_t move_reduction;
    move_reduction.resize(send_count);

    bin_t particles_in_bins;
    for (int i = 0; i < particles_positions.size(); ++i) {
      particles_in_bins.insert(particles_in_bins.end(),
                               particles_positions[i].begin(), particles_positions[i].end());
    }

    MPI_Scatterv(particles_in_bins.data(), receive_count, displace, PARTICLE,
                 move_reduction.data(), send_count, PARTICLE, 0, MPI_COMM_WORLD);

    int l = 0;
    while (l < send_count){
      particle_t &p = move_reduction[i];
      assert(p.x >= 0 && p.y >= 0 && p.x <= grid_size && p.y <= grid_size);
      push_particles_in_bin(p, bins);

      ++l;
    }
//    for (int i = 0; i < send_count; ++i) {
//      particle_t &p = move_reduction[i];
//      assert(p.x >= 0 && p.y >= 0 && p.x <= grid_size && p.y <= grid_size);
//      push_particles_in_bin(p, bins);
//    }
    }
    simulation_time = read_timer() - simulation_time;

    if (rank == 0) {
      printf("n = %d, simulation time = %g seconds", n, simulation_time);

      if (find_option(argc, argv, "-no") == -1) {
        if (nabsavg) absavg /= nabsavg;
        //
        //  -The minimum distance absmin between 2 particles during the run of the simulation
        //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
        //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
        //
        //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
        //
        printf(", absmin = %lf, absavg = %lf", absmin, absavg);
        if (absmin < 0.4) printf("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
        if (absavg < 0.8) printf("\nThe average distance is below 0.8 meaning that most particles are not interacting");
      }
      printf("\n");

      //
      // Printing summary data
      //
      if (fsum)
        fprintf(fsum, "%d %d %g\n", n, n_proc, simulation_time);
    }

    //
    //  release resources
    //
    if (fsum)
      fclose(fsum);
    if (fsave)
      fclose(fsave);

    MPI_Finalize();

    return 0;
  }
