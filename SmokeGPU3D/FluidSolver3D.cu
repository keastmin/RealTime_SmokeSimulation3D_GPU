#include "FluidSolver3D.cuh"

#define IX(i, j, k) ((i) + (N+2)*(j) + (N+2)*(N+2)*(k))
#define SWAP(x0, x) {double* tmp=x0;x0=x;x=tmp;}
#define LINEARSOLVERTIMES 20


// 소스항 커널 함수
__global__ void add_source(int N, double* x, double* s, double dt) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int size = (N + 2) * (N + 2) * (N + 2);
	if (idx < size) {
		x[idx] += dt * s[idx];
	}
}

/* --------------------경계조건 설정-------------------- */
// 면에 대한 커널 함수
__global__ void faces_bnd(int N, int b, double*x) {
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
	if (i <= N && j <= N) {
		x[IX(i, j, 0)] = b == 3 ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
		x[IX(i, j, N + 1)] = b == 3 ? -x[IX(i, j, N)] : x[IX(i, j, N)];
		x[IX(0, i, j)] = b == 1 ? -x[IX(1, i, j)] : x[IX(1, i, j)];
		x[IX(N + 1, i, j)] = b == 1 ? -x[IX(N, i, j)] : x[IX(N, i, j)];
		x[IX(i, 0, j)] = b == 2 ? -x[IX(i, 1, j)] : x[IX(i, 1, j)];
		x[IX(i, N + 1, j)] = b == 2 ? -x[IX(i, N, j)] : x[IX(i, N, j)];
	}
}

// x축 엣지에 대한 커널 함수
__global__ void x_edge_bnd(int N, double* x) {
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (i <= N) {
		x[IX(i, 0, 0)] = 1.0 / 2.0 * (x[IX(i, 1, 0)] + x[IX(i, 0, 1)]);
		x[IX(i, N + 1, 0)] = 1.0 / 2.0 * (x[IX(i, N, 0)] + x[IX(i, N + 1, 1)]);
		x[IX(i, 0, N + 1)] = 1.0 / 2.0 * (x[IX(i, 0, N)] + x[IX(i, 1, N + 1)]);
		x[IX(i, N + 1, N + 1)] = 1.0 / 2.0 * (x[IX(i, N, N + 1)] + x[IX(i, N + 1, N)]);
	}
}
// y축 엣지에 대한 커널 함수
__global__ void y_edge_bnd(int N, double* x) {
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (i <= N) {
		x[IX(0, i, 0)] = 1.0 / 2.0 * (x[IX(1, i, 0)] + x[IX(0, i, 1)]);
		x[IX(N + 1, i, 0)] = 1.0 / 2.0 * (x[IX(N, i, 0)] + x[IX(N + 1, i, 1)]);
		x[IX(0, i, N + 1)] = 1.0 / 2.0 * (x[IX(0, i, N)] + x[IX(1, i, N + 1)]);
		x[IX(N + 1, i, N + 1)] = 1.0 / 2.0 * (x[IX(N, i, N + 1)] + x[IX(N + 1, i, N)]);
	}
}
// z축 엣지에 대한 커널 함수
__global__ void z_edge_bnd(int N, double* x) {
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (i <= N) {
		x[IX(0, 0, i)] = 1.0 / 2.0 * (x[IX(0, 1, i)] + x[IX(1, 0, i)]);
		x[IX(0, N + 1, i)] = 1.0 / 2.0 * (x[IX(0, N, i)] + x[IX(1, N + 1, i)]);
		x[IX(N + 1, 0, i)] = 1.0 / 2.0 * (x[IX(N, 0, i)] + x[IX(N + 1, 1, i)]);
		x[IX(N + 1, N + 1, i)] = 1.0 / 2.0 * (x[IX(N + 1, N, i)] + x[IX(N, N + 1, i)]);
	}
}

// 코너에 대한 커널 함수
__global__ void corner_bnd(int N, double* x) {
	x[IX(0, 0, 0)] = 1.0 / 3.0 * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
	x[IX(0, N + 1, 0)] = 1.0 / 3.0 * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);

	x[IX(N + 1, 0, 0)] = 1.0 / 3.0 * (x[IX(N, 0, 0)] + x[IX(N + 1, 1, 0)] + x[IX(N + 1, 0, 1)]);
	x[IX(N + 1, N + 1, 0)] = 1.0 / 3.0 * (x[IX(N, N + 1, 0)] + x[IX(N + 1, N, 0)] + x[IX(N + 1, N + 1, 1)]);

	x[IX(0, 0, N + 1)] = 1.0 / 3.0 * (x[IX(1, 0, N + 1)] + x[IX(0, 1, N + 1)] + x[IX(0, 0, N)]);
	x[IX(0, N + 1, N + 1)] = 1.0 / 3.0 * (x[IX(1, N + 1, N + 1)] + x[IX(0, N, N + 1)] + x[IX(0, N + 1, N)]);

	x[IX(N + 1, 0, N + 1)] = 1.0 / 3.0 * (x[IX(N, 0, N + 1)] + x[IX(N + 1, 1, N + 1)] + x[IX(N + 1, 0, N)]);
	x[IX(N + 1, N + 1, N + 1)] = 1.0 / 3.0 * (x[IX(N, N + 1, N + 1)] + x[IX(N + 1, N, N + 1)] + x[IX(N + 1, N + 1, N)]);
}

// 경계조건 커널 구동 함수
void set_bnd(int N, int b, double* x) {
	// 면에 대한 경계조건
	dim3 blockDim(16, 16);
	dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
	faces_bnd<<<gridDim, blockDim>>>(N, b, x);

	// 엣지에 대한 경계조건
	int blockSize = 256;
	int numBlock = (N + blockSize - 1) / blockSize;
	x_edge_bnd<<<numBlock, blockSize>>> (N, x); // x축 엣지
	y_edge_bnd<<<numBlock, blockSize>>> (N, x); // y축 엣지
	z_edge_bnd<<<numBlock, blockSize>>> (N, x); // z축 엣지

	// 코너에 대한 경계조건
	corner_bnd<<<1, 1>>>(N, x);
}
/* ---------------------------------------------------- */

/* ------------선형방정식 red black gauss seidel------------ */
// red 셀 커널
__global__ void red_cell_lin(int N, double* x, double* x0, double a, double c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

	if (i <= N && j <= N && k <= N && (i + j + k) % 2 == 0) {
		x[IX(i, j, k)] = (x0[IX(i, j, k)] + a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] + x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] + x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) / c;
	}
}

// black 셀 커널
__global__ void black_cell_lin(int N, double* x, double* x0, double a, double c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

	if (i <= N && j <= N && k <= N && (i + j + k) % 2 != 0) {
		x[IX(i, j, k)] = (x0[IX(i, j, k)] + a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] + x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] + x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) / c;
	}
}

// 커널함수 구동
void lin_solve(int N, int b, double* x, double* x0, double a, double c) {
	int l;
	dim3 blockDim(8, 8, 8);
	dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y, (N + blockDim.z - 1) / blockDim.z);

	for (l = 0; l < LINEARSOLVERTIMES; l++) {
		red_cell_lin<<<gridDim, blockDim>>>(N, x, x0, a, c);
		cudaDeviceSynchronize();

		black_cell_lin<<<gridDim, blockDim>>>(N, x, x0, a, c);
		cudaDeviceSynchronize();

		set_bnd(N, b, x);
		cudaDeviceSynchronize();
	}
}
/* -------------------------------------------------------- */

// 확산 함수
void diffuse(int N, int b, double* x, double* x0, double diff, double dt) {
	double a = dt * diff * N * N * N;
	lin_solve(N, b, x, x0, a, 1 + 6 * a);
}

/* ------------------------이류 함수------------------------ */
// advect 커널 함수 정의
__global__ void k_advect(int N, double* d, double* d0, double* u, double* v, double* w, double dt) {
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
	if (i <= N && j <= N && k <= N) {
		int i0, j0, k0, i1, j1, k1;
		double x, y, z, s0, t0, s1, t1, u1, u0, dtx, dty, dtz;
		dtx = dty = dtz = dt * N;
		x = i - dtx * u[IX(i, j, k)]; y = j - dty * v[IX(i, j, k)]; z = k - dtz * w[IX(i, j, k)];
		if (x < 0.5f) x = 0.5f; if (x > N + 0.5f) x = N + 0.5f; i0 = (int)x; i1 = i0 + 1;
		if (y < 0.5f) y = 0.5f; if (y > N + 0.5f) y = N + 0.5f; j0 = (int)y; j1 = j0 + 1;
		if (z < 0.5f) z = 0.5f; if (z > N + 0.5f) z = N + 0.5f; k0 = (int)z; k1 = k0 + 1;

		s1 = x - i0; s0 = 1 - s1; t1 = y - j0; t0 = 1 - t1; u1 = z - k0; u0 = 1 - u1;


		d[IX(i, j, k)] = s0 * (t0 * u0 * d0[IX(i0, j0, k0)] + t1 * u0 * d0[IX(i0, j1, k0)] + t0 * u1 * d0[IX(i0, j0, k1)] + t1 * u1 * d0[IX(i0, j1, k1)]) +
			s1 * (t0 * u0 * d0[IX(i1, j0, k0)] + t1 * u0 * d0[IX(i1, j1, k0)] + t0 * u1 * d0[IX(i1, j0, k1)] + t1 * u1 * d0[IX(i1, j1, k1)]);
	}
}

// advect 커널 구동 함수
void advect(int N, int b, double* d, double* d0, double* u, double* v, double* w, double dt) {
	dim3 blockDim(8, 8, 8);
	dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y, (N + blockDim.z - 1) / blockDim.z);
	k_advect<<<gridDim, blockDim>>>(N, d, d0, u, v, w, dt);
	cudaDeviceSynchronize();

	set_bnd(N, b, d);
	cudaDeviceSynchronize();
}
/* -------------------------------------------------------- */

/* ----------------------프로젝트 함수---------------------- */
// 발산을 계산하고 압력 필드를 0으로 초기화
__global__ void calcDiv(int N, double*u, double*v, double*w, double* p, double* div) {
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
	double h = 1.0 / N;
	if (i <= N && j <= N && k <= N) {
		div[IX(i, j, k)] = -0.5 * h * (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] +
			v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)] +
			w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]);
		p[IX(i, j, k)] = 0;
	}
}

// 속도 필드 업데이트 (질량 보존)
__global__ void massConserve(int N, double* u, double* v, double* w, double* p) {
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
	double h = 1.0 / N;
	if (i <= N && j <= N && k <= N) {
		u[IX(i, j, k)] -= 0.5 * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]) / h;
		v[IX(i, j, k)] -= 0.5 * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]) / h;
		w[IX(i, j, k)] -= 0.5 * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]) / h;
	}
}

// 프로젝트 커널함수를 구동하는 함수
void project(int N, double* u, double* v, double* w, double* p, double* div) {
	dim3 blockDim(8, 8, 8);
	dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y, (N + blockDim.z - 1) / blockDim.z);
	calcDiv<<<gridDim, blockDim>>> (N, u, v, w, p, div);
	cudaDeviceSynchronize();

	set_bnd(N, 0, div); set_bnd(N, 0, p);
	cudaDeviceSynchronize();

	lin_solve(N, 0, p, div, 1, 6);

	massConserve<<<gridDim, blockDim>>>(N, u, v, w, p);
	cudaDeviceSynchronize();

	set_bnd(N, 1, u); set_bnd(N, 2, v); set_bnd(N, 3, w);
	cudaDeviceSynchronize();
}
/* -------------------------------------------------------- */

// 밀도 필드 업데이트
void dens_step(int N, double* x, double* x0, double* u, double* v, double* w, double diff, double dt) {
	// 소스항 추가
	int size = (N + 2) * (N + 2) * (N + 2);
	int blockSize = 256;
	int numBlocks = (size + blockSize - 1) / blockSize;
	add_source<<<numBlocks, blockSize>>>(N, x, x0, dt);
	cudaDeviceSynchronize();

	SWAP(x0, x); diffuse(N, 0, x, x0, diff, dt);
	SWAP(x0, x); advect(N, 0, x, x0, u, v, w, dt);
}


// 속도 필드 업데이트
void vel_step(int N, double* u, double* v, double* w, double* u0, double* v0, double* w0, double visc, double dt) {
	// 소스항 추가
	int size = (N + 2) * (N + 2) * (N + 2);
	int blockSize = 256;
	int numBlocks = (size + blockSize - 1) / blockSize;
	add_source<<<numBlocks, blockSize>>>(N, u, u0, dt);
	add_source<<<numBlocks, blockSize>>>(N, v, v0, dt);
	add_source<<<numBlocks, blockSize>>>(N, w, w0, dt);
	cudaDeviceSynchronize();

	// 스왑 후 확산항
	SWAP(u0, u); diffuse(N, 1, u, u0, visc, dt);
	SWAP(v0, v); diffuse(N, 2, v, v0, visc, dt);
	SWAP(w0, w); diffuse(N, 3, w, w0, visc, dt);

	// 프로젝션
	project(N, u, v, w, u0, v0);

	// 이류
	SWAP(u0, u); SWAP(v0, v); SWAP(w0, w);
	advect(N, 1, u, u0, u0, v0, w0, dt);
	advect(N, 2, v, v0, u0, v0, w0, dt);
	advect(N, 3, w, w0, u0, v0, w0, dt);

	// 마지막 프로젝션
	project(N, u, v, w, u0, v0);
}