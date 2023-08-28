#include "control.h"

glm::mat4 ProjectionMatrix;
glm::mat4 ViewMatrix;

glm::mat4 getProjectionMatrix() {
	return ProjectionMatrix;
}

glm::mat4 getViewMatrix() {
	return ViewMatrix;
}


// 좌표
glm::vec3 position = glm::vec3(0.5, 0.5, 2);

// 수평각
float horizontalAngle = 3.14;
// 수직각
float verticalAngle = 0.0f;
// 초기 FOV값
float initialFoV = 45.0f;

float speed = 2.5f;
float mouseSpeed = 0.05f;

void init_position() {
	position = glm::vec3(0.5, 0.5, 2);
	horizontalAngle = 3.14;
	verticalAngle = 0.0f;
}

void computeMatricesFromInputs(GLFWwindow* window, int width, int height) {
	static double lastTime = glfwGetTime();

	double currentTime = glfwGetTime();
	float deltaTime = float(currentTime - lastTime);

	// 마우스 좌표 받기
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	glfwSetCursorPos(window, width / 2, height / 2);

	// 새로운 방향 계산
	horizontalAngle += mouseSpeed * deltaTime * double(width / 2 - xpos);
	verticalAngle += mouseSpeed * deltaTime * double(height / 2 - ypos);

	// direction : 구면 좌표를 데카르트 좌표로 변환
	glm::vec3 direction = glm::vec3(
		cos(verticalAngle) * sin(horizontalAngle),
		sin(verticalAngle),
		cos(verticalAngle) * cos(horizontalAngle)
	);

	// right 벡터
	glm::vec3 right(
		sin(horizontalAngle - 3.14f / 2.0f),
		0,
		cos(horizontalAngle - 3.14f / 2.0f)
	);

	// up 벡터 : Direction과 Right에 대해 직각
	glm::vec3 up = glm::cross(right, direction);

	// 앞으로 이동
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		position += direction * deltaTime * speed;
	}
	// 뒤로 이동
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		position -= direction * deltaTime * speed;
	}
	// 오른쪽으로 이동
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		position += right * deltaTime * speed;
	}
	// 왼쪽으로 이동
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		position -= right * deltaTime * speed;
	}
	// 위로 이동
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
		position += up * deltaTime * speed;
	}
	// 아래로 이동
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
		position -= up * deltaTime * speed;
	}
	
	float FoV = initialFoV;

	ProjectionMatrix = glm::perspective(FoV, (float)(width / height), 0.1f, 100.0f);

	ViewMatrix = glm::lookAt(
		position,
		position + direction,
		up
	);

	lastTime = currentTime;
}
