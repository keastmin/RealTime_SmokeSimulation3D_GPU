#ifndef __CONTROL_H__
#define __CONTROL_H__

#include <iostream>
#include <math.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

void computeMatricesFromInputs(GLFWwindow* window, int width, int height);
void init_position();
glm::mat4 getProjectionMatrix();
glm::mat4 getViewMatrix();

#endif __CONTROL_H__