#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include "omp.h"
#include <vector>
using namespace std;

#define _cutoff 0.01
#define _density 0.0005
double bin, grid;
int binNumber;

void buildBin(vector<bin_t>& bins, particle_t* particle, int j)
{
    grid = sqrt(n*_density);
    bin = _cutoff * 2;
    binNumber = int(grid/bin) + 1;
    bins.resize(binNumber * binNumber);
    for(int i = 0; i < j; i++)
    {
        int x = int(particle[i].x / bin);
        int y = int(particle[i].y / bin);
        bins[x * binNumber + y].push_back(particle[i]);
    }
}

//
//  benchmarking program
//
int main( int argc, char **argv )
{   
    int navg,nabsavg=0,numthreads; 
    double dmin, absmin=1.0,davg,absavg=0.0;
	
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" ); 
        printf( "-no turns off all correctness checks and particle output\n");   
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;      

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    set_size( n );
    init_particles( n, particles );

    vector<bin_t> particleBin;
    vector<bin_t> temp;

    buildBin(particleBin, particles, n);

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );

    #pragma omp parallel private(dmin) 
    {
        #pragma omp master
        {
            numthreads = omp_get_num_threads();
            temp.resize(numthreads);
        };

        for( int step = 0; step < 1000; step++ )
        {
            navg = 0;
            davg = 0.0;
            dmin = 1.0;
            //
            //  compute all forces
            //
            #pragma omp for reduction (+:navg) reduction(+:davg)
            for( int i = 0; i < binNumber; i++ )
            {
                for (int j = 0; j < binNumber; j++ )
                {
                    bin_t& vecBin = particleBin[i * binNumber + j];

                    for(int k = 0; k < vecBin.size(); k++)
                    {
                        vecBin[k].ax = vecBin[k].ay = 0.0;
                    }

                    for(int bx = -1; bx <= 1; bx++)
                    {
                        for(int by = -1; by <= 1; by++)
                        {
                            if(i + bx >= 0 && i + bx < binNumber && j + by >= 0 && j + by < binNumber)
                            {
                                bin_t& vect = particleBin[(i + bx) * binNumber + j + by];
                                for(int l = 0; l < vect.size(); l++)
                                {
                                    for(int m = 0; m < vect.size(); m++)
                                    {
                                        apply_force(vecBin[k], vect[m], &dmin, &davg, &navg);
                                    }
                                }
                            }
                        }
                    }
                }
            }


            //
            //  move particles
            //
            int threadId = omp_get_thread_num();
            bin_t& clean = clean[threadId];
            clean.clear();
            #pragma omp for
            for(int i = 0; i < binNumber; i++ )
            {
                for(int j = 0; j < binNumber; j++)
                {
                    bin_t& vec = particleBin[i * binNumber + j];
                    int tailRec = vec.size();
                    int k = 0;
                    for(;k < tailRec;)
                    {
                        move(vec[k]);
                        int x = int(vec[k].x / bin);
                        int y = int(vec[k].y / bin);
                        if(x == i && y == j)
                        {
                            k++;
                        }
                        else
                        {
                            clean.push_back(vec[k]);
                            vec[k] = vec[--tailRec];
                        }
                    }
                    vec.resize(k);
                }
            }
            #pragma omp master
            {
                for(int i = 0; i < numthreads; i++)
                {
                    bin_t& temp2 = temp[i];
                    for(int j = 0; j < temp2.size(); j++)
                    {
                        int x = int(temp2[j].x / bin);
                        int y = int(temp2[j].y / bin);
                        particleBin[x * binNumber + y].push_back(temp2[j]);
                    }
                }
            }

            if( find_option( argc, argv, "-no" ) == -1 )
            {
                 //
                 //  compute statistical data
                 //
                 #pragma omp master
                 if (navg) {
                   absavg += davg/navg;
                   nabsavg++;
                 }

                 #pragma omp critical
                 if (dmin < absmin) absmin = dmin;

                  //
                  //  save if necessary
                  //
                 #pragma omp master
                 if( fsave && (step%SAVEFREQ) == 0 )
                     save( fsave, n, particles );
            }
            #pragma omp barrier
        }
}
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "n = %d,threads = %d, simulation time = %g seconds", n,numthreads, simulation_time);

    if( find_option( argc, argv, "-no" ) == -1 )
    {
      if (nabsavg) absavg /= nabsavg;
    // 
    //  -the minimum distance absmin between 2 particles during the run of the simulation
    //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
    //  -A simulation were particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
    //
    //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
    //
    printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
    if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
    if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
    }
    printf("\n");
    
    //
    // Printing summary data
    //
    if( fsum)
        fprintf(fsum,"%d %d %g\n",n,numthreads,simulation_time);

    //
    // Clearing space
    //
    if( fsum )
        fclose( fsum );

    free( particles );
    if( fsave )
        fclose( fsave );
    
    return 0;
}
