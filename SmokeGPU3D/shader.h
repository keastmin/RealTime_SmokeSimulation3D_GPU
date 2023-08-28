#ifndef __SHADER_H__
#define __SHADER_H__

#include <iostream>
#include <fstream>
#include <cstring>
#include <sstream>
#include <vector>
#include "GL/glew.h"
#include "GLFW/glfw3.h"

GLuint LoadShaders(const char* vertex_file_path, const char* fragment_file_path);

#endif __SHADER_H__