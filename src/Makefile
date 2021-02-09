# If USE_SURF is true, the software necessary to adjust camera motion with
# SURF features is included.
# Note that this requires building OpenCV 3.x with opencv_contrib and
# turning on OPENCV_ENABLE_NONFREE.
USE_SURF = true

# If USE_GPROF is true, adds options for gprof and turns off printing.
#USE_GPROF = true

# If USE_PYTHON is true, a library using the Python C API is built.
#USE_PYTHON = true

# The directory containing the Python include directory.
PREFIX=${HOME}/py-virt
# The Python version for include and library files.
PYTHON_VER=3.5m

# set the binaries that have to be built
ifeq ($(USE_PYTHON),true)
TARGETS := DenseTrackStab.so
USE_GPROF = false
else
TARGETS := DenseTrackStab Video
endif

# set the build configuration set 
BUILD := release
#BUILD := debug

# set bin and build dirs
BUILDDIR := .build_$(BUILD)
BINDIR := $(BUILD)

# libraries 
LDLIBS = $(addprefix -l, $(LIBS) $(LIBS_$(notdir $*)))
LIBS := opencv_videoio opencv_calib3d opencv_features2d opencv_highgui opencv_imgproc opencv_imgcodecs opencv_core
ifeq ($(USE_SURF),true)
LIBS += opencv_xfeatures2d
endif
ifeq ($(USE_PYTHON),true)
LIBS += python${PYTHON_VER}
endif

# set some flags and compiler/linker specific commands
CXXFLAGS = -pipe -D __STDC_CONSTANT_MACROS -D STD=std -Wall $(CXXFLAGS_$(BUILD)) -I. -I/opt/include
ifeq ($(USE_SURF),true)
CXXFLAGS += -DUSE_SURF
endif
ifeq ($(USE_PYTHON),true)
CXXFLAGS += -DUSE_PYTHON -I${PREFIX}/include/python${PYTHON_VER}
endif
ifeq ($(USE_GPROF),true)
CXXFLAGS += -DUSE_GPROF -pg
endif

CXXFLAGS_debug := -ggdb
CXXFLAGS_release := -O3 -DNDEBUG -ggdb
LDFLAGS = -L/opt/lib -pipe -Wall $(LDFLAGS_$(BUILD))
ifeq ($(USE_GPROF),true)
LDFLAGS += -pg
endif
LDFLAGS_debug := -ggdb
LDFLAGS_release := -O3 -ggdb

include make/generic.mk
