#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// 컨트롤, 쉐이더, 솔버
#include "control.h"
#include "shader.h"
#include "FluidSolver3D.cuh"

// 3차원 배열을 1차원 배열처럼 관리하기 위한 IX 정의
#define IX(i, j, k) ((i) + (N+2)*(j) + (N+2)*(N+2)*(k))

// 그리드 사이즈 : SIZE * SIZE * SIZE (3차원)
#define SIZE 50

// 윈도우 크기
#define WIDTH 800
#define HEIGHT 800

/* ----------솔버에 사용될 데이터 선언---------- */
// cudaMalloc으로 초기화되어 커널에 사용될 데이터
static double* u, * v, * w, * u_prev, * v_prev, * w_prev;
static double* dens, * dens_prev;

// 상수 데이터
static const int N = SIZE;
static double dt = 0.1f;
static double diff = 0.0f;
static double visc = 0.0f;
static double force = 20.0f;
static double source = 100.0f;
static double source_alp = 0.03f;
/* -------------------------------------------- */

// 윈도우 크기 정의
const static int width = WIDTH;
const static int height = HEIGHT;

// mode == 0 : 연기, mode == 1 : 속도장
static int mode = 0;

// addforce == 0 : 외력 없음, addforce == 1 : 외력 추가
static int addforce = 0;

/* -----------CUDA to OpenGL 버텍스, 컬러 버퍼----------- */
// 밀도의 고정 위치 버퍼
static glm::vec3* dens_buffer;
// 밀도의 색상 버퍼
static glm::vec3* dens_color_buffer;
/* --------------------- */

// 속도장의 고정 위치 버퍼
static glm::vec3* static_vel_buffer;
// 속도장의 이동 위치 버퍼
static glm::vec3* dynamic_vel_buffer;
// 속도장의 라인을 그릴 버퍼
static glm::vec3* vel_buffer;
/* ----------------------------------------------------- */

// GLFW 윈도우 선언
GLFWwindow* window;

// 메모리 해제
void free_data() {
	if (u) cudaFree(u);
	if (v) cudaFree(v);
	if (v) cudaFree(w);
	if (u_prev) cudaFree(u_prev);
	if (v_prev) cudaFree(v_prev);
	if (w_prev) cudaFree(w_prev);
	if (dens) cudaFree(dens);
	if (dens_prev) cudaFree(dens_prev);
	if (static_vel_buffer) cudaFree(static_vel_buffer);
	if (dynamic_vel_buffer) cudaFree(dynamic_vel_buffer);
}

/* ------------------------초기화------------------------ */
// 커널 데이터 삽입 함수
__global__ void initArray(double* array, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		array[i] = 0.0;
	}
}

// 데이터 초기화 함수
static int init_data() {
	int size;
	size_t d_size;
	cudaError_t err;

	// 솔버 데이터 메모리 할당
	size = (N + 2) * (N + 2) * (N + 2);
	d_size = size * sizeof(double);
	err = cudaMalloc((void**)&u, d_size);
	if (err != cudaSuccess) {
		std::cerr << "Error allocating u : " << cudaGetErrorString(err) << '\n';
		free_data();
		return 0;
	}
	err = cudaMalloc((void**)&v, d_size);
	if (err != cudaSuccess) {
		std::cerr << "Error allocating v : " << cudaGetErrorString(err) << '\n';
		free_data();
		return 0;
	}
	err = cudaMalloc((void**)&w, d_size);
	if (err != cudaSuccess) {
		std::cerr << "Error allocating w : " << cudaGetErrorString(err) << '\n';
		free_data();
		return 0;
	}
	err = cudaMalloc((void**)&u_prev, d_size);
	if (err != cudaSuccess) {
		std::cerr << "Error allocating u_prev : " << cudaGetErrorString(err) << '\n';
		free_data();
		return 0;
	}
	err = cudaMalloc((void**)&v_prev, d_size);
	if (err != cudaSuccess) {
		std::cerr << "Error allocating v_prev : " << cudaGetErrorString(err) << '\n';
		free_data();
		return 0;
	}
	err = cudaMalloc((void**)&w_prev, d_size);
	if (err != cudaSuccess) {
		std::cerr << "Error allocating w_prev : " << cudaGetErrorString(err) << '\n';
		free_data();
		return 0;
	}
	err = cudaMalloc((void**)&dens, d_size);
	if (err != cudaSuccess) {
		std::cerr << "Error allocating dens : " << cudaGetErrorString(err) << '\n';
		free_data();
		return 0;
	}
	err = cudaMalloc((void**)&dens_prev, d_size);
	if (err != cudaSuccess) {
		std::cerr << "Error allocating dens_prev : " << cudaGetErrorString(err) << '\n';
		free_data();
		return 0;
	}

	// 속도장 라인 데이터 할당
	size = (N + 2) * (N + 2) * (N + 2);
	d_size = size * sizeof(glm::vec3);
	err = cudaMalloc((void**)&static_vel_buffer, d_size);
	if (err != cudaSuccess) {
		std::cerr << "Error allocating static_vel_buffer : " << cudaGetErrorString(err) << '\n';
		free_data();
		return 0;
	}
	err = cudaMalloc((void**)&dynamic_vel_buffer, d_size);
	if (err != cudaSuccess) {
		std::cerr << "Error allocating dunamic_vel_buffer : " << cudaGetErrorString(err) << '\n';
		free_data();
		return 0;
	}

	// 솔버 데이터 초기화
	size = (N + 2) * (N + 2) * (N + 2);
	int blockSize = 256;
	int numBlocks = (size + blockSize - 1) / blockSize;
	initArray<<<numBlocks, blockSize>>>(u, size);
	initArray<<<numBlocks, blockSize>>>(v, size);
	initArray<<<numBlocks, blockSize>>>(w, size);
	initArray<<<numBlocks, blockSize>>>(u_prev, size);
	initArray<<<numBlocks, blockSize>>>(v_prev, size);
	initArray<<<numBlocks, blockSize>>>(w_prev, size);
	initArray<<<numBlocks, blockSize>>>(dens, size);
	initArray<<<numBlocks, blockSize>>>(dens_prev, size);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "Filed to launch initArray kernel : " << cudaGetErrorString(err) << '\n';
		free_data();
		return 0;
	}

	return 1;
}
/* ------------------------------------------------------ */

// 키보드 콜백 함수
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_Z && action == GLFW_RELEASE) {
		addforce = (addforce == 0) ? 1 : 0;
		std::cout << "addforce : " << addforce << '\n';
	}

	if (key == GLFW_KEY_1 && action == GLFW_RELEASE) {
		mode = 0;
		std::cout << "mode : " << mode << '\n';
	}

	if (key == GLFW_KEY_2 && action == GLFW_RELEASE) {
		mode = 1;
		std::cout << "mode : " << mode << '\n';
	}
}

/* --------------------외력 추가-------------------- */
// 외력 추가 커널 함수
__global__ void set_force_source(double* d, int di, int dj, int dk, double dV, double* f, int fi, int fj, int fk, double fV) {
	int fIdx = IX(fi, fj, fk);
	int dIdx = IX(di, dj, dk);
	f[fIdx] = fV;
	d[dIdx] = dV;
}

// 데이터 초기화 및 외력 위치 지정
void add_force_source(double* d, double* u, double* v, double* w) {
	int i, j, k, size = (N + 2) * (N + 2) * (N + 2);
	cudaMemset(u, 0, size * sizeof(double));
	cudaMemset(v, 0, size * sizeof(double));
	cudaMemset(w, 0, size * sizeof(double));
	cudaMemset(d, 0, size * sizeof(double));
	cudaDeviceSynchronize();

	if (addforce == 1) {
		i = N / 2;
		j = 2;
		k = N / 2;

		if (i < 1 || i > N || j < 1 || j > N || k < 1 || k > N) {
			std::cerr << "범위 벗어남" << '\n';
			return;
		}
		double forceValue = force * 3;
		double sourceValue = source;
		set_force_source<<<1, 1>>>(d, i, j + 3, k, sourceValue, v, i, j, k, forceValue);
		
		//i = N / 2;
		//j = N / 2;
		//k = N - 3;
		//set_force_source << <1, 1 >> > (d, i, j + 3, k, sourceValue, v, i, j, k, forceValue);
	}
}
/* ------------------------------------------------ */

/* -----------velocity 초기화 및 업데이트 함수----------- */
// velocity 초기화 커널 함수
__global__ void init_vel(int hN, glm::vec3* vel, glm::vec3* stvel, glm::vec3* dyvel) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (i < hN && j < hN && k < hN) {
		int idx = IX(i, j, k);
		double x, y, z, h;
		h = 1.0 / hN;
		x = ((i + 1) - 0.5) * h;
		y = ((j + 1) - 0.5) * h;
		z = ((k + 1) - 0.5) * h;
		
		stvel[idx].x = x;
		stvel[idx].y = y;
		stvel[idx].z = z;

		dyvel[idx].x = x;
		dyvel[idx].y = y;
		dyvel[idx].z = z;

		vel[2 * idx] = stvel[idx];
		vel[2 * idx + 1] = dyvel[idx];		
	}
}

// velocity 업데이트 커널 함수
__global__ void update_vel(glm::vec3* vel, int hN, double* ku, double* kv, double* kw, glm::vec3* dyvel) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	int idx = IX(i, j, k);
	int velIdx = IX(i + 1, j + 1, k + 1);
	if (i < hN && j < hN && k < hN) {
		vel[2 * idx + 1].x = dyvel[idx].x + ku[velIdx];
		vel[2 * idx + 1].y = dyvel[idx].y + kv[velIdx];
		vel[2 * idx + 1].z = dyvel[idx].z + kw[velIdx];		
	}
}
/* ---------------------------------------------------- */

/* ------------density 초기화 및 업데이트 함수------------ */
// 정육면체 면을 채우는 함수
__device__ void addCubeFaceDevice(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec3* buffer, int& index) {
	buffer[index++] = p0;
	buffer[index++] = p1;
	buffer[index++] = p2;

	buffer[index++] = p2;
	buffer[index++] = p3;
	buffer[index++] = p0;
}

// density 초기화 커널 함수
__global__ void init_dens(int hN, glm::vec3* dens) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	int idx = IX(i, j, k);
	double h = 1.0 / hN;
	if (i < hN && j < hN && k < hN) {
		float x = ((i + 1) - 0.5f) * h;
		float y = ((j + 1) - 0.5f) * h;
		float z = ((k + 1) - 0.5f) * h;

		glm::vec3 p000(x, y, z);
		glm::vec3 p100(x + h, y, z);
		glm::vec3 p110(x + h, y + h, z);
		glm::vec3 p101(x + h, y, z + h);
		glm::vec3 p111(x + h, y + h, z + h);
		glm::vec3 p010(x, y + h, z);
		glm::vec3 p011(x, y + h, z + h);
		glm::vec3 p001(x, y, z + h);

		int localIdx = idx * 36;
		addCubeFaceDevice(p000, p010, p110, p100, dens, localIdx);
		addCubeFaceDevice(p001, p011, p111, p101, dens, localIdx);
		addCubeFaceDevice(p000, p001, p101, p100, dens, localIdx);
		addCubeFaceDevice(p010, p011, p111, p110, dens, localIdx);
		addCubeFaceDevice(p000, p010, p011, p001, dens, localIdx);
		addCubeFaceDevice(p100, p110, p111, p101, dens, localIdx);
	}
}

// density color 초기화 커널 함수
__global__ void init_dens_color(int hN, glm::vec3* cDens) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	int idx = IX(i, j, k);
	if (i < hN && j < hN && k < hN) {
		int localIdx = 36 * idx;
		glm::vec3 icolor(0.0f, 0.0f, 0.0f);
		addCubeFaceDevice(icolor, icolor, icolor, icolor, cDens, localIdx);
		addCubeFaceDevice(icolor, icolor, icolor, icolor, cDens, localIdx);
		addCubeFaceDevice(icolor, icolor, icolor, icolor, cDens, localIdx);
		addCubeFaceDevice(icolor, icolor, icolor, icolor, cDens, localIdx);
		addCubeFaceDevice(icolor, icolor, icolor, icolor, cDens, localIdx);
		addCubeFaceDevice(icolor, icolor, icolor, icolor, cDens, localIdx);
	}
}

// density color 업데이트 커널 함수
__global__ void update_dens_color(int hN, glm::vec3* cDens, double* kdens){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	double d000, d100, d110, d101, d111, d010, d011, d001;
	int idx = IX(i, j, k);
	if (i < hN && j < hN && k < hN) {
		d000 = kdens[IX(i + 1, j + 1, k + 1)];
		d100 = kdens[IX(i + 2, j + 1, k + 1)];
		d110 = kdens[IX(i + 2, j + 2, k + 1)];
		d101 = kdens[IX(i + 2, j + 1, k + 2)];
		d111 = kdens[IX(i + 2, j + 2, k + 2)];
		d010 = kdens[IX(i + 1, j + 2, k + 1)];
		d011 = kdens[IX(i + 1, j + 2, k + 2)];
		d001 = kdens[IX(i + 1, j + 1, k + 2)];

		glm::vec3 p000(d000, d000, d000);
		glm::vec3 p100(d100, d100, d100);
		glm::vec3 p110(d110, d110, d110);
		glm::vec3 p101(d101, d101, d101);
		glm::vec3 p111(d111, d111, d111);
		glm::vec3 p010(d010, d010, d010);
		glm::vec3 p011(d011, d011, d011);
		glm::vec3 p001(d001, d001, d001);

		//glm::vec3 p000(d000 + 0.7, d000, d000);
		//glm::vec3 p100(d100 + 0.7, d100, d100);
		//glm::vec3 p110(d110 + 0.7, d110, d110);
		//glm::vec3 p101(d101 + 0.7, d101, d101);
		//glm::vec3 p111(d111 + 0.7, d111, d111);
		//glm::vec3 p010(d010 + 0.7, d010, d010);
		//glm::vec3 p011(d011 + 0.7, d011, d011);
		//glm::vec3 p001(d001 + 0.7, d001, d001);

		int localIdx = 36 * idx;
		addCubeFaceDevice(p000, p010, p110, p100, cDens, localIdx);
		addCubeFaceDevice(p001, p011, p111, p101, cDens, localIdx);
		addCubeFaceDevice(p000, p001, p101, p100, cDens, localIdx);
		addCubeFaceDevice(p010, p011, p111, p110, cDens, localIdx);
		addCubeFaceDevice(p000, p010, p011, p001, cDens, localIdx);
		addCubeFaceDevice(p100, p110, p111, p101, cDens, localIdx);
	}
}

/* ----------------------------------------------------- */

// 시뮬레이션 함수
void sim_fluid() {
	add_force_source(dens_prev, u_prev, v_prev, w_prev);
	vel_step(N, u, v, w, u_prev, v_prev, w_prev, visc, dt);
	dens_step(N, dens, dens_prev, u, v, w, diff, dt);
	cudaDeviceSynchronize();
}

int main() {
	// GLFW 초기화
	if (!glfwInit()) {
		std::cerr << "GLFW 초기화 실패" << '\n';
		glfwTerminate();
		return -1;
	}
	glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing 설정
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); // OpenGL 4.x 버전 사용
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5); // OpenGL 4.5 버전 사용
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // Core Profile 사용
	window = glfwCreateWindow(width, height, "SmokeSimulatuin3D", NULL, NULL);
	if (window == NULL) {
		std::cerr << "GLFW 초기화 실패" << '\n';
		glfwTerminate();
		return -1;
	}

	// GLEW 초기화
	glfwMakeContextCurrent(window); // 현제 윈도우에서 OpenGL 작업이 이루어짐
	glewExperimental = true; // OpenGL 확장 기능 효율적으로 가져옴
	if (glewInit() != GLEW_OK) {
		std::cerr << "GLEW 초기화 실패" << '\n';
		glfwTerminate();
		return -1;
	}

	// 커널 데이터 초기화
	if (!init_data()) {
		std::cerr << "메모리 할당 실패" << '\n';
		glfwTerminate();
		return -1;
	}
	cudaDeviceSynchronize();

	// 마우스 커서 세팅
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // 커서 숨기기
	glfwPollEvents();
	glfwSetCursorPos(window, width / 2, height / 2); // 커서 위치(중앙)

	// Shader 읽기
	GLuint programID = LoadShaders("VertexShaderSL.txt", "FragmentShaderSL.txt");
	GLuint MatrixID = glGetUniformLocation(programID, "MVP");
	GLuint velColor = glGetUniformLocation(programID, "colorMode");
	GLuint alpValue = glGetUniformLocation(programID, "alphaValue");

	// 사용할 그래픽 카드
	cudaSetDevice(0);

	// VAO
	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	// 커널함수 구동 그리드 사이즈
	dim3 blockDim(8, 8, 8);
	dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y, (N + blockDim.z - 1) / blockDim.z);
	
	/* ----------------속도장 표현 버퍼 정의---------------- */
	GLuint velocitybuffer;
	glGenBuffers(1, &velocitybuffer);
	glBindBuffer(GL_ARRAY_BUFFER, velocitybuffer);
	glBufferData(GL_ARRAY_BUFFER, 2 * (N + 2) * (N + 2) * (N + 2) * sizeof(glm::vec3), NULL, GL_STREAM_DRAW);
	
	cudaGraphicsResource* cudaVBOVel;
	size_t numByteVel;
	cudaGraphicsGLRegisterBuffer(&cudaVBOVel, velocitybuffer, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsMapResources(1, &cudaVBOVel, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&vel_buffer, &numByteVel, cudaVBOVel);
	init_vel<<<gridDim, blockDim>>>(N, vel_buffer, static_vel_buffer, dynamic_vel_buffer);
	cudaDeviceSynchronize();
	cudaGraphicsUnmapResources(1, &cudaVBOVel, 0);
	/* --------------------------------------------------- */

	/* ----------------밀도장 표현 버퍼 정의---------------- */
	GLuint densitybuffer;
	glGenBuffers(1, &densitybuffer);
	glBindBuffer(GL_ARRAY_BUFFER, densitybuffer);
	glBufferData(GL_ARRAY_BUFFER, 36 * (N + 2) * (N + 2) * (N + 2) * sizeof(glm::vec3), NULL, GL_STATIC_DRAW);

	cudaGraphicsResource* cudaVBODens;
	size_t numByteDens;
	cudaGraphicsGLRegisterBuffer(&cudaVBODens, densitybuffer, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsMapResources(1, &cudaVBODens, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&dens_buffer, &numByteDens, cudaVBODens);
	
	init_dens<<<gridDim, blockDim>>>(N, dens_buffer);
	cudaDeviceSynchronize();
	cudaGraphicsUnmapResources(1, &cudaVBODens, 0);

	GLuint densitycolorbuffer;
	glGenBuffers(1, &densitycolorbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, densitycolorbuffer);
	glBufferData(GL_ARRAY_BUFFER, 36 * (N + 2) * (N + 2) * (N + 2) * sizeof(glm::vec3), NULL, GL_STREAM_DRAW);
	
	cudaGraphicsResource* cudaVBODensColor;
	size_t numByteDensColor;
	cudaGraphicsGLRegisterBuffer(&cudaVBODensColor, densitycolorbuffer, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsMapResources(1, &cudaVBODensColor, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&dens_color_buffer, &numByteDensColor, cudaVBODensColor);

	init_dens_color<<<gridDim, blockDim>>>(N, dens_color_buffer);
	cudaDeviceSynchronize();
	cudaGraphicsUnmapResources(1, &cudaVBODensColor, 0);
	/* --------------------------------------------------- */

	// 키보드 콜백 함수 호출
	glfwSetKeyCallback(window, key_callback);
	// 솔버 데이터 업데이트 및 그리기 (while문 ESC로 종료)
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE); // Stick Keys 활성화
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	do {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glUseProgram(programID);
		
		// 화면 조작 (control.h)
		computeMatricesFromInputs(window, width, height);
		glm::mat4 ProjectionMatrix = getProjectionMatrix();
		glm::mat4 ViewMatrix = getViewMatrix();
		glm::mat4 ModelMatrix = glm::mat4(1.0);
		glm::mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;
		glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

		// 시뮬레이션 반복
		sim_fluid();

		if (mode == 0) {
			glUniform1i(velColor, GL_FALSE);
			glUniform1d(alpValue, source_alp);

			cudaGraphicsMapResources(1, &cudaVBODensColor, 0);
			cudaGraphicsResourceGetMappedPointer((void**)&dens_color_buffer, &numByteDensColor, cudaVBODensColor);
			update_dens_color<<<gridDim, blockDim >>>(N, dens_color_buffer, dens);
			cudaDeviceSynchronize();
			cudaGraphicsUnmapResources(1, &cudaVBODensColor, 0);

			glBindBuffer(GL_ARRAY_BUFFER, densitybuffer);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(
				0,
				3,
				GL_FLOAT,
				GL_FALSE,
				0,
				(void*)0
			);

			glBindBuffer(GL_ARRAY_BUFFER, densitycolorbuffer);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(
				1,
				3,
				GL_FLOAT,
				GL_FALSE,
				0,
				(void*)0
			);

			glDrawArrays(GL_TRIANGLES, 0, 36 * (N + 2) * (N + 2) * (N + 2));

			glDisableVertexAttribArray(0);
			glDisableVertexAttribArray(1);
		}
		
		if (mode == 1) {
			glUniform1i(velColor, GL_TRUE);

			cudaGraphicsMapResources(1, &cudaVBOVel, 0);
			cudaGraphicsResourceGetMappedPointer((void**)&vel_buffer, &numByteVel, cudaVBOVel);
			update_vel<<<gridDim, blockDim>>>(vel_buffer, N, u, v, w, dynamic_vel_buffer);
			cudaDeviceSynchronize();
			cudaGraphicsUnmapResources(1, &cudaVBOVel, 0);
			
			glBindBuffer(GL_ARRAY_BUFFER, velocitybuffer);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(
				0,
				3,
				GL_FLOAT,
				GL_FALSE,
				0,
				(void*)0
			);

			glDrawArrays(GL_LINES, 0, 2 * (N + 2) * (N + 2) * (N + 2));

			glDisableVertexAttribArray(0);
		}

		glfwSwapBuffers(window);
		glfwPollEvents();
	} while ((glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0));
	
	// 데이터 정리
	glDeleteProgram(programID);
	glDeleteVertexArrays(1, &VertexArrayID);
	glDeleteBuffers(1, &velocitybuffer);
	glDeleteBuffers(1, &densitybuffer);
	glDeleteBuffers(1, &densitycolorbuffer);
	glfwDestroyWindow(window);
	free_data();
	glfwTerminate();
	
	return 0;
}