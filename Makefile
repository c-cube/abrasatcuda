include config

#--------------------------------------------------------------
# predefined vars, and real makefile vars
#--------------------------------------------------------------

#path to the cuda compiler
# TODO : complete this accroding to the computer's configuration (autoconf)
CUDAPATH = /usr/local
CUDA_INCLUDES = -I/usr/local/cuda-2.3/include/


# current compiler
CC = gcc
NVCC = ${CUDAPATH}/bin/nvcc

# some vars
SRC = src
DIST = dist
LIB = lib

# for cuda flags
NVFLAGS = -O2
ifeq ($(EMUU),yes)
	NVFLAGS += -deviceemu
endif

CFLAGS = -Wall -pedantic -Os -std=gnu99 -DPARALLEL=$(PARALLEL) -fPIC #-m32 -Werror
#Variable contenant les options passées au compilateur
DBG = -g
ifeq ($(DEBUG),yes)
	DBG = -g -DDEBUG=1
endif
# NDEBUG disables all assert() statements, so it accelerates the program
ifeq ($(DEBUG),prod)
	DBG = -DNDEBUG=1
	NVFLAGS += --compiler-options -fno-strict-aliasing
endif
ifeq ($(DEBUG),all)
	DBG = -g -DDEBUG=2
endif


#for cuda compilation
CUDA=
ifeq ($(PARALLEL),cuda)
	CUDA = -DCUDA=1 -DNDEBUG=1 -DDEBUG=0
endif

PROF=
ifeq ($(PROFILE),yes)
	PROF = -pg
endif
#L'option -Wall affiche tous les messages d'alertes (warnings)
#L'option -Werror traite une simple alerte comme une erreur (stoppant ainsi lq compilation)
#L'option -std= permet de fixer la norme ISO du C que le compilateur va utiliser pour vérifier la validité du programme.
#L'option -pedantic empeche l'utilisation des extensions et impose de se conformer à la version stricte de la norme ISO
#L'option -O permet de définir le degré d'optimisation appliqué par le compilateur (0 est la valeur par défaut, aucune optimisation)
#L'option -g compile le programme en laissant les informations nécessaires pour l'utilisation d'un debugger lors de l'exécution.
#L'option -m indique que le code généré doit être pour un environement 32 bits
#L'option -c indique que le compilateur doit se contenter de la génération d'un fichier objet. La création d'un exécutable se fera si nécessaire dans un second temps.

# in case you prefer compile the program as a dynamic lib, this allows to use local .so files
# TODO[find sth better] : export LD_LIBRARY_PATH := .


# lists of targets, headers, objets files...
PLUGINS = single pthread
BINARIES = abrasatcuda
ifeq ($(COMPILE_CUDA),yes)
	BINARIES += abrasatcuda_cuda
	PLUGINS += cuda
endif

TARGETS = $(addsuffix .so, $(addprefix $(LIB)/,$(PLUGINS))) $(addprefix $(DIST)/,$(BINARIES))

BASE_OBJECTS = clause parser heuristic
OBJECTS = $(addsuffix .o, $(addprefix $(SRC)/,$(BASE_OBJECTS)))

BASE_MODULES = dpll brute_force single_thread
MODULES = $(addsuffix .o, $(addprefix $(SRC)/, $(BASE_MODULES)))

BASE_HEADERS = list clause parser abrasatcuda interfaces dpll vars consts brute_force interfaces/dispatch heuristic
HEADERS = $(addsuffix .h, $(addprefix $(SRC)/, $(BASE_HEADERS)))

BASE_COMMON_OBJECTS = dpll
COMMON_OBJECTS = $(addsuffix .o, $(addprefix $(SRC)/,$(BASE_COMMON_OBJECTS)))

# flags for dynamic libs
DYNFLAGS = -shared

#--------------------------------------------------------------
# targets
#--------------------------------------------------------------

# default target
all: $(TARGETS) $(MODULES)


# launches tests
test: ${DIST}/test_all
	./${DIST}/test_all

main: $(TARGETS)
	@echo -e "\n\e[45;4mexample.cnf :\e[m"
	@./abrasatcuda tests/example.cnf
	@echo -e "\n\e[45;4mtrivial.cnf :\e[m"
	@./abrasatcuda tests/trivial.cnf
	@echo -e "\n\e[45;4mfalse.cnf :\e[m"
	@./abrasatcuda tests/false.cnf
	@echo -e "\n\e[45;4mquinn.cnf :\e[m"
	@./abrasatcuda tests/quinn.cnf
	@echo -e "\n\e[45;4maim-50.cnf :\e[m"
	@./abrasatcuda tests/aim-50-1_6-yes1-4.cnf


prof:
	gprof ./abrasatcuda


check: ${SRC}/check.hs
	ghc -O2 --make ${SRC}/check.hs -o ${DIST}/check

count:
	@echo "number of code/comment lines : "; grep -v '^[ ]*$$' ./${SRC}/{*.h,*.c} | wc -l


# This targets compiles the main binary
${DIST}/abrasatcuda: ${SRC}/abrasatcuda.c $(OBJECTS)
	$(CC) $(LDFLAGS) $(CFLAGS) $(DBG) $(PROF) -DTHREAD_NUM=${THREAD_NUM} $(OBJECTS) ${SRC}/abrasatcuda.c -ldl -lm -pthread -o ${DIST}/abrasatcuda

%.so: $(COMMON_OBJECTS) $(OBJECTS)

${LIB}/single.so: $(COMMON_OBJECTS) $(OBJECTS) ${SRC}/single_thread.o
	$(CC) $(LDFLAGS) $(CFLAGS) $(DBG) $(PROF) $(OBJECTS) $(COMMON_OBJECTS) -DPARALLEL=single ${SRC}/single_thread.o ${SRC}/abrasatcuda.c $(DYNFLAGS) -o ${LIB}/single.so

${LIB}/pthread.so: $(COMMON_OBJECTS) $(OBJECTS) ${SRC}/multi_thread.o
	$(CC) $(LDFLAGS) $(CFLAGS) $(DBG) $(PROF) $(OBJECTS) $(COMMON_OBJECTS) -DPARALLEL=pthread ${SRC}/multi_thread.o  ${SRC}/abrasatcuda.c $(DYNFLAGS) -o ${LIB}/pthread.so

${LIB}/cuda.so: $(COMMON_OBJECTS) $(OBJECTS) $(DISPATCH_OBJECT) $(SRC)/cuda.o
	$(CC) $(LDFLAGS) $(CUDA_INCLUDES) $(NVFLAGS) -L$(CUDA_INCLUDES) $(PROF) $(CUDA) $(OBJECTS)  $(DYNFLAGS) ${SRC}/cuda.o ${SRC}/abrasatcuda.c -lcudart -o $(LIB)/cuda.so

${DIST}/abrasatcuda_cuda: $(OBJECTS) $(DISPATCH_OBJECT) $(SRC)/cuda.o $(SRC)/abrasatcuda.c
	$(CC) $(LDFLAGS) $(CUDA_INCLUDES) $(NVFLAGS) -L$(CUDA_INCLUDES) $(PROF) $(CUDA) $(OBJECTS) ${SRC}/cuda.o ${SRC}/abrasatcuda.c  -lcudart -o $(DIST)/abrasatcuda_cuda

#
# object files
%.o: %.c %.h
	$(CC) $(CFLAGS) $(DBG) $(PROF) -c $< -o $@

${SRC}/dpll.o: ${SRC}/interfaces/solve.h

${SRC}/brute_force.o: ${SRC}/interfaces/solve.h

${SRC}/single_thread.o: ${SRC}/interfaces/solve.h

${SRC}/multi_thread.o: ${SRC}/interfaces/solve.h

${SRC}/cuda.o: ${SRC}/solve.cu ${SRC}/dpll_while.c $(SRC)/heuristic.c $(SRC)/solve.h
	$(NVCC)  $(CUDA_INCLUDES) $(NVFLAGS) ${SRC}/solve.cu  $(PROF) $(CUDA) -c -o ${SRC}/cuda.o



#Cette cible effectue un simple nettoyage des fichiers temporaires qui ont pu être générés
clean:
	@rm -f ${SRC}/*.o ${LIB}/* ${DIST}/*
	@rm -f test_all

#Cette cible effectue un nettoyage complet de tout fichier généré. Elle efface notamment les exécutables.
distclean: clean
	@rm -f $(TARGETS)
