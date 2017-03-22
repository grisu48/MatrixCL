
# This is the path where your AMD OpenCL SDK is located
AMDSDK_PATH=/opt/AMDAPP
#/opt/intel-opencl-sdk
#/opt/AMDAPPSDK
#/opt/AMDAPP

# ==== COMPILER FLAGS ==== #
O_FLAGS=-Wall -Wextra -pedantic -g -O0
#O_FLAGS=-Wall -Wextra -pedantic -O3
CXX=g++
CXX_FLAGS=$(O_FLAGS) -std=c++11


# ==== INCLUDE AND LIB ==== #
INCLUDE=-I$(AMDSDK_PATH)/include -L$(AMDSDK_PATH)/lib/x86_64 -I../Matrix -I../util -L../Matrix/Linux-amd64 -L../util/Linux-amd64
OCL_LIBS=-lOpenCL
# GLUT libs, needed if the library is compiled with OpenGL support
GLUT_LIBS=-lm -lGLU -lglut -lGLU -lGL -lGLEW
OBJS=FlexCL.o FlexCLMatrix.o
LIBS=$(OCL_LIBS) $(GLUT_LIBS)

# ==== User configuration ===== #
USER_LIBS=
# Binaries that should be produced
BINS=cl_matrix





# ==== Makefile section ==== #

default:	all
all:	$(OBJS) $(BINS)
clean:
	rm -f *.o


# ==== PROGRAM SECTION ===== #
# Insert your program instructuions here


cl_matrix:	CL_Matrix.cpp $(OBJS)
	$(CXX) $(CXX_FLAGS) -o $@ $< $(OBJS) $(INCLUDE) $(LIBS) -pthread $(USER_LIBS)


# ==== Additional objects ==== #

%.o:	%.cpp %.hpp
	$(CXX) $(CXX_FLAGS) -c -o $@ $< $(INCLUDE)
