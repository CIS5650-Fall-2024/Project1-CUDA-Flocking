/**
 * @file      main.cpp
 * @brief     Example Boids flocking simulation for CIS 5650
 * @authors   Liam Boone, Kai Ninomiya, Kangning (Gary) Li
 * @date      2013-2017
 * @copyright University of Pennsylvania
 */

#include "main.hpp"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "kernel.h"
#include <numeric>
#include <deque>

// ================
// Configuration
// ================

// LOOK-2.1 LOOK-2.3 - toggles for UNIFORM_GRID and COHERENT_GRID
#define VISUALIZE 1
#define UNIFORM_GRID 1
#define COHERENT_GRID 1

// LOOK-1.2 - change this to adjust particle count in the simulation
// const int N_FOR_VIS = 5000;
const float DT = 0.2f;

/**
 * C main function.
 */
int main(int argc, char *argv[])
{
  projectName = "5650 CUDA Intro: Boids";
  if (!init(argc, argv)) {
    return -1;
  }

  // printBenchmarks();
  mainLoop();
  Boids::endSimulation();
  return 0;
}

void printBenchmarks() {
  std::array<unsigned int, 4> boidCounts = {1000, 10000, 50000, 100000};
  std::array<void(*)(float), 3> simulations = {Boids::stepSimulationNaive, Boids::stepSimulationScatteredGrid, Boids::stepSimulationCoherentGrid, };
  std::array<std::string, 3> simNames = {"Naive", "Scattered", "Coherent"};

  for (unsigned int blockSize = 16; blockSize <= 1024; blockSize *= 4) {
    for (unsigned int numBoids : boidCounts) {
      // auto err = cudaGetLastError();
      // if (cudaGetLastError() != err) {
      //   std::cerr << "Cuda error: " << err << std::endl;
      //   exit(1);
      // }
      fillAndRegisterOpenGLBuffers(numBoids);
      Boids::initSimulation(numBoids, blockSize);
      std::cerr << blockSize << " " << numBoids << " ";
      for (size_t i = 0; i < 3; i++) {
        if (numBoids >= 50000 && i == 0) {
          std::cerr << "N/A ";
          continue;
        }
        double benchmarkMs = benchmarkMsPerFrame(numBoids, simulations[i]);
        std::cerr << benchmarkMs << " ";
      }
      std::cerr << std::endl;
      Boids::endSimulation();
      cudaDeviceSynchronize();
      unregisterOpenGLBuffers();
    }
  }
  
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

std::string deviceName;
GLFWwindow *window;

/**
* Initialization of CUDA and GLFW.
*/
bool init(int argc, char **argv) {
  // Set window title to "Student Name: [SM 2.0] GPU Name"
  cudaDeviceProp deviceProp;
  int gpuDevice = 0;
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (gpuDevice > device_count)
  {
    std::cout
        << "Error: GPU device number is greater than the number of devices!"
        << " Perhaps a CUDA-capable GPU is not installed?"
        << std::endl;
    return false;
  }
  cudaGetDeviceProperties(&deviceProp, gpuDevice);
  int major = deviceProp.major;
  int minor = deviceProp.minor;

  std::ostringstream ss;
  ss << projectName << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
  deviceName = ss.str();

  // Window setup stuff
  glfwSetErrorCallback(errorCallback);

  if (!glfwInit())
  {
    std::cout
        << "Error: Could not initialize GLFW!"
        << " Perhaps OpenGL 3.3 isn't available?"
        << std::endl;
    return false;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
  if (!window)
  {
    glfwTerminate();
    return false;
  }
  glfwMakeContextCurrent(window);
  glfwSetKeyCallback(window, keyCallback);
  glfwSetCursorPosCallback(window, mousePositionCallback);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);

  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK)
  {
    return false;
  }
  // Default to device ID 0. If you have more than one GPU and want to test a non-default one,
  // change the device ID.
  cudaGLSetGLDevice(0);

  initVAO();

  updateCamera();

  initShaders(program);

  glEnable(GL_DEPTH_TEST);

  return true;
}

void initVAO() {
  glGenVertexArrays(1, &boidVAO); // Attach everything needed to draw a particle to this
  glGenBuffers(1, &boidVBO_positions);
  glGenBuffers(1, &boidVBO_velocities);

  glBindVertexArray(boidVAO);

  // Bind the positions array to the boidVAO by way of the boidVBO_positions
  glBindBuffer(GL_ARRAY_BUFFER, boidVBO_positions); // bind the buffer

  glEnableVertexAttribArray(positionLocation);
  glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

  // Bind the velocities array to the boidVAO by way of the boidVBO_velocities
  glBindBuffer(GL_ARRAY_BUFFER, boidVBO_velocities);
  glEnableVertexAttribArray(velocitiesLocation);
  glVertexAttribPointer((GLuint)velocitiesLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);
}

void fillAndRegisterOpenGLBuffers(unsigned int numBoids) {
  std::vector<glm::vec4> bodies(numBoids, glm::vec4(0, 0, 0, 1));
  glBindBuffer(GL_ARRAY_BUFFER, boidVBO_positions);
  glBufferData(GL_ARRAY_BUFFER, numBoids * sizeof(glm::vec4), bodies.data(), GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, boidVBO_velocities);
  glBufferData(GL_ARRAY_BUFFER, numBoids * sizeof(glm::vec4), bodies.data(), GL_DYNAMIC_DRAW);
  cudaGLRegisterBufferObject(boidVBO_positions);
  cudaGLRegisterBufferObject(boidVBO_velocities);  
}

void unregisterOpenGLBuffers()
{
  cudaGLUnregisterBufferObject(boidVBO_positions);
  cudaGLUnregisterBufferObject(boidVBO_velocities);
}

void initShaders(GLuint *program)
{
  GLint location;

  program[PROG_BOID] = glslUtility::createProgram(
      "shaders/boid.vert.glsl",
      "shaders/boid.geom.glsl",
      "shaders/boid.frag.glsl", attributeLocations, 2);
  glUseProgram(program[PROG_BOID]);

  if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1)
  {
    glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
  }
  if ((location = glGetUniformLocation(program[PROG_BOID], "u_cameraPos")) != -1)
  {
    glUniform3fv(location, 1, &cameraPosition[0]);
  }
}

//====================================
// Main loop
//====================================
void runCUDA(void(*simulation)(float))
{
  // Map OpenGL buffer object for writing from CUDA on a single GPU
  // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not
  // use this buffer

  float4 *dptr = NULL;
  float *dptrVertPositions = NULL;
  float *dptrVertVelocities = NULL;

  cudaGLMapBufferObject((void **)&dptrVertPositions, boidVBO_positions);
  cudaGLMapBufferObject((void **)&dptrVertVelocities, boidVBO_velocities);

  simulation(DT);

#if VISUALIZE
  Boids::copyBoidsToVBO(dptrVertPositions, dptrVertVelocities);
#endif
  // unmap buffer object
  cudaGLUnmapBufferObject(boidVBO_positions);
  cudaGLUnmapBufferObject(boidVBO_velocities);
}

// Gets #microseconds per frame
// Waits until we have a converged time (variance over the past dequeSize frames is < thresholdVariance)
// Or until 5 minutes have passed
double benchmarkMsPerFrame(unsigned int numBoids, void (*simulation)(float))
{
  const size_t dequeSize = 1024;
  const double desiredRuntime = 120;

  double startTime = glfwGetTime();
  double timebase;

  std::deque<double> pastFrames;

  double mean = 0;

  while (true) {
    glfwPollEvents();

    if (glfwWindowShouldClose(window)) {
      glfwDestroyWindow(window);
      glfwTerminate();
      return mean;
    }

    timebase = glfwGetTime();
    runCUDA(simulation);
    double time = glfwGetTime();
    double frameTime = time - timebase;
    pastFrames.push_back(frameTime * 1000);
    if (pastFrames.size() > dequeSize) {
      pastFrames.pop_front();
    }
    
    mean = std::accumulate(pastFrames.begin(), pastFrames.end(), 0.0) / pastFrames.size();

    double runningTime = time - startTime;
    if (runningTime > desiredRuntime) {
      break;
    }

    std::ostringstream ss;
    ss << "ms per frame: ";
    ss.precision(5);
    ss << mean;
    glfwSetWindowTitle(window, ss.str().c_str());
  }

  return mean;
}

void mainLoop() {
  void(*simulation)(float);
  #if UNIFORM_GRID && COHERENT_GRID
    simulation = Boids::stepSimulationCoherentGrid;
  #elif UNIFORM_GRID
    simulation = Boids::stepSimulationScatteredGrid;
  #else
    simulation = Boids::stepSimulationNaive;
  #endif

  unsigned int numBoids = 50000;
  unsigned int blockSize = 128;
  fillAndRegisterOpenGLBuffers(numBoids);
  Boids::initSimulation(numBoids, blockSize);

  double fps = 0;
  double timebase = 0;
  int frame = 0;

  Boids::unitTest(); // LOOK-1.2 We run some basic example code to make sure
                     // your CUDA development setup is ready to go.

  while (!glfwWindowShouldClose(window))
  {
    glfwPollEvents();

    frame++;
    double time = glfwGetTime();

    if (time - timebase > 1.0)
    {
      fps = frame / (time - timebase);
      timebase = time;
      frame = 0;
    }

    runCUDA(simulation);

    std::ostringstream ss;
    ss << "[";
    ss.precision(1);
    ss << std::fixed << fps;
    ss << " fps] " << deviceName;
    glfwSetWindowTitle(window, ss.str().c_str());

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      #if VISUALIZE
      glUseProgram(program[PROG_BOID]);
      glBindVertexArray(boidVAO);
      glPointSize((GLfloat)pointSize);
      glDrawArrays(GL_POINTS, 0, numBoids);
      glPointSize(1.0f);

    glUseProgram(0);
    glBindVertexArray(0);

    glfwSwapBuffers(window);
#endif
  }
  glfwDestroyWindow(window);
  glfwTerminate();
}

void errorCallback(int error, const char *description)
{
  fprintf(stderr, "error %d: %s\n", error, description);
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
  {
    glfwSetWindowShouldClose(window, GL_TRUE);
  }
}

void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
  leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
  rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow *window, double xpos, double ypos)
{
  if (leftMousePressed)
  {
    // compute new camera parameters
    phi += (xpos - lastX) / width;
    theta -= (ypos - lastY) / height;
    theta = std::fmax(0.01f, std::fmin(theta, 3.14f));
    updateCamera();
  }
  else if (rightMousePressed)
  {
    zoom += (ypos - lastY) / height;
    zoom = std::fmax(0.1f, std::fmin(zoom, 5.0f));
    updateCamera();
  }

  lastX = xpos;
  lastY = ypos;
}

void updateCamera()
{
  cameraPosition.x = zoom * sin(phi) * sin(theta);
  cameraPosition.z = zoom * cos(theta);
  cameraPosition.y = zoom * cos(phi) * sin(theta);
  cameraPosition += lookAt;

  projection = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
  glm::mat4 view = glm::lookAt(cameraPosition, lookAt, glm::vec3(0, 0, 1));
  projection = projection * view;

  GLint location;

  glUseProgram(program[PROG_BOID]);
  if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1)
  {
    glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
  }
}
