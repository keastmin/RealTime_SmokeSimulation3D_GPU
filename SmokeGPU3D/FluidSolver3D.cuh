#ifndef __FLUIDSOLVER3D_H__
#define __FLUIDSOLVER3D_H__

#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void add_source(int N, double* x, double* s, double dt);
void set_bnd(int N, int b, double* x);
void lin_solve(int N, int b, double* x, double* x0, double a, double c);
void diffuse(int N, int b, double* x, double* x0, double diff, double dt);
void advect(int N, int b, double* d, double* d0, double* u, double* v, double* w, double dt);
void project(int N, double* u, double* v, double* w, double* p, double* div);
void dens_step(int N, double* x, double* x0, double* u, double* v, double* w, double diff, double dt);
void vel_step(int N, double* u, double* v, double* w, double* u0, double* v0, double* w0, double visc, double dt);

#endif __FLUIDSOLVER3D_h__