#include "control.h"

glm::mat4 ProjectionMatrix;
glm::mat4 ViewMatrix;

glm::mat4 getProjectionMatrix() {
	return ProjectionMatrix;
}

glm::mat4 getViewMatrix() {
	return ViewMatrix;
}


// ��ǥ
glm::vec3 position = glm::vec3(0.5, 0.5, 2);

// ����
float horizontalAngle = 3.14;
// ������
float verticalAngle = 0.0f;
// �ʱ� FOV��
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

	// ���콺 ��ǥ �ޱ�
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	glfwSetCursorPos(window, width / 2, height / 2);

	// ���ο� ���� ���
	horizontalAngle += mouseSpeed * deltaTime * double(width / 2 - xpos);
	verticalAngle += mouseSpeed * deltaTime * double(height / 2 - ypos);

	// direction : ���� ��ǥ�� ��ī��Ʈ ��ǥ�� ��ȯ
	glm::vec3 direction = glm::vec3(
		cos(verticalAngle) * sin(horizontalAngle),
		sin(verticalAngle),
		cos(verticalAngle) * cos(horizontalAngle)
	);

	// right ����
	glm::vec3 right(
		sin(horizontalAngle - 3.14f / 2.0f),
		0,
		cos(horizontalAngle - 3.14f / 2.0f)
	);

	// up ���� : Direction�� Right�� ���� ����
	glm::vec3 up = glm::cross(right, direction);

	// ������ �̵�
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		position += direction * deltaTime * speed;
	}
	// �ڷ� �̵�
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		position -= direction * deltaTime * speed;
	}
	// ���������� �̵�
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		position += right * deltaTime * speed;
	}
	// �������� �̵�
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		position -= right * deltaTime * speed;
	}
	// ���� �̵�
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
		position += up * deltaTime * speed;
	}
	// �Ʒ��� �̵�
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
