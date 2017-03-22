
/* =============================================================================
 *
 * Title:         MatrixCL
 * Author:        Felix Niederwanger
 * Description:   Standalone matrix framework with OpenCL backend
 *
 * =============================================================================
 */

#ifndef MARIXCL_HPP_
#define MARIXCL_HPP_



#include <iostream>
#include <vector>
#include <string>
#include <exception>


#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


namespace matrixcl {

class OpenCLException;
class IOException;
class DeviceException;
class CompileException;

/* OpenCL-based exception */
class OpenCLException : public std::exception {
protected:
    /** Error message. */
    std::string _msg;

    cl_int err_code;
public:
    explicit OpenCLException(const char* msg, cl_int err_code = 0):_msg(std::string(msg)) { this->err_code = err_code; }
    explicit OpenCLException(std::string msg, cl_int err_code = 0):_msg(msg) { this->err_code = err_code; }
    explicit OpenCLException(cl_int err_code = 0):_msg("") { this->err_code = err_code; }
    virtual ~OpenCLException() throw () {}

    /** @return error message */
    std::string getMessage() { return _msg; }
    /** @return error message */
    virtual const char* what() const throw () {
    	return _msg.c_str();
    }
    /** @return returned opencl error code  */
    cl_int error_code(void) { return this->err_code; }

    /** @return is the exception has an error code */
    bool hasErrorCode(void) { return this->err_code != 0; }

    /** @return OpenCL error code message as a string */
    virtual std::string opencl_error_string(void) {
		switch(err_code) {
			case CL_SUCCESS:                            return "Success";
			case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
			case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
			case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
			case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
			case CL_OUT_OF_RESOURCES:                   return "Out of resources";
			case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
			case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
			case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
			case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
			case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
			case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
			case CL_MAP_FAILURE:                        return "Map failure";
			case CL_INVALID_VALUE:                      return "Invalid value";
			case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
			case CL_INVALID_PLATFORM:                   return "Invalid platform";
			case CL_INVALID_DEVICE:                     return "Invalid device";
			case CL_INVALID_CONTEXT:                    return "Invalid context";
			case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
			case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
			case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
			case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
			case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
			case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
			case CL_INVALID_SAMPLER:                    return "Invalid sampler";
			case CL_INVALID_BINARY:                     return "Invalid binary";
			case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
			case CL_INVALID_PROGRAM:                    return "Invalid program";
			case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
			case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
			case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
			case CL_INVALID_KERNEL:                     return "Invalid kernel";
			case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
			case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
			case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
			case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
			case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
			case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
			case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
			case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
			case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
			case CL_INVALID_EVENT:                      return "Invalid event";
			case CL_INVALID_OPERATION:                  return "Invalid operation";
			case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
			case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
			case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
			default:									return "";
		}
	}
};


/** Exception that is associated to a device access  */
class DeviceException : public OpenCLException {
public:
    explicit DeviceException(std::string msg, cl_int err_code = 0) : OpenCLException(msg, err_code) {}
    explicit DeviceException(const char* msg, cl_int err_code = 0) : OpenCLException(msg, err_code) {}
    virtual ~DeviceException() {}
};

/** Exception when processing a IO operations  */
class IOException : public OpenCLException {
public:
    explicit IOException(std::string msg, cl_int err_code = 0) : OpenCLException(msg, err_code) {}
    explicit IOException(const char* msg, cl_int err_code = 0) : OpenCLException(msg, err_code) {}
    virtual ~IOException() {}
};

/** OpenCL compile exception  */
class CompileException : public OpenCLException {
protected:
	const char* source;
	size_t length;

	cl_device_id *_device_id;
	std::string _compile_output;

public:
    explicit CompileException(std::string msg, cl_device_id *device_id, std::string compile_output, cl_int err_code = 0) : OpenCLException(msg, err_code) {
		this->_device_id = device_id;
		this->_compile_output = compile_output;
		this->length = 0;
		this->source = "";
	}
    virtual ~CompileException() throw () {}

    cl_device_id* device_id() { return this->_device_id; }
    std::string compile_output() { return this->_compile_output; }
};




/** OpenCL context */
class ContextCL {
protected:
	/** Underlying actual OpenCL context */
	cl_context context;

	ContextCL(cl_context context);
public:
	/** Copy constructor */
	ContextCL(const ContextCL &context);
	virtual ~ContextCL();
};


/** Matrix template */
template<class T>
class Matrix {
protected:
	/** Assigned OpenCL context */
	ContextCL* context;


	/** OpenCL memory for this matrix */
	cl_mem mem;

	/** Total memory size */
	size_t size;

	/** RIM cells in each dimension */
	size_t _rim;

	/** Optional a matrix can hold a name */
	std::string _name;

	/** Resize the matrix, i.e. a new memory is assigned.
	 * Due to performance considerations the buffer is not cleared, so the contents are undefined
	 * */
	void resize(const size_t size);
	/** Clear current buffer, i.e. replace it with zeros */
	void clear();
	/** Set all elements to the given value */
	void set(const T &val);

	/** Assign values to this matrix
	 * If the size mismatched with the current matrix size, previously a resize will be called
	 * */
	void set(const T* src, size_t size);

public:
	Matrix();
	virtual ~Matrix();

	/** @returns the name of the matrix */
	std::string name(void) const { return this->_name; }
	/** Set the name of the matrix
	 * @param name to be assigned
	 */
	void setName(const std::string &name) { this->_name = name; }

};

/** 2d Matrix */
template<class T>
class Matrix2d : public Matrix<T> {
protected:
	/** Defined number of matrix cells without RIM cells */
	size_t _mx[2];

public:
	Matrix2d(const size_t mx, const size_t my);
	Matrix2d(const size_t mx[2]);
	Matrix2d(const Matrix2d &src);
	virtual ~Matrix2d();

	void resize(const size_t mx, const size_t my);
	void resize(const size_t mx, const size_t my, const size_t rim);

	T& get(const size_t ix, const size_t iy);
	T& operator()(const size_t ix, const size_t iy);


};

}



#endif /* MARIXCL_HPP_ */
