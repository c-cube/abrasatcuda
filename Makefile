include config

#--------------------------------------------------------------
# predefined vars, and real makefile vars
#--------------------------------------------------------------

#path to the cuda compiler
# TODO : complete this accroding to the computer's configuration
CUDAPATH=/usr/local
CUDA_INCLUDES=-I /usr/local/cuda-2.3/include/


# current compiler
CC=gcc
NVCC=${CUDAPATH}/bin/nvcc

# some vars
SRC=src
BUILD=build
DIST=dist
LIB=lib

# for cuda flags
NVFLAGS=-O2
ifeq ($(EMUU),yes)
NVFLAGS += -deviceemu
endif

CFLAGS=-Wall -pedantic -Os -std=gnu99 -DPARALLEL=$(PARALLEL) -fPIC #-m32 -Werror
#Variable contenant les options passées au compilateur
DBG=-g
ifeq ($(DEBUG),yes)
	DBG=-g -DDEBUG=1 
endif
# NDEBUG disables all assert() statements, so it accelerates the program
ifeq ($(DEBUG),prod)
	DBG=-DNDEBUG=1 
	NVFLAGS += --compiler-options -fno-strict-aliasing
endif
ifeq ($(DEBUG),all)
	DBG=-g -DDEBUG=2
endif


#for cuda compilation
CUDA=
ifeq ($(PARALLEL),cuda)
	CUDA=-DCUDA=1 -DNDEBUG=1 -DDEBUG=0
endif

PROF=
ifeq ($(PROFILE),yes)
	PROF=-pg
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
TARGETS=${DIST}/abrasatcuda_bf ${LIB}/abrasatcuda_dpll_single.so ${LIB}/abrasatcuda_dpll_pthread.so ${DIST}/abrasatcuda #${DIST}/abrasatcuda_cuda
OBJECTS=${BUILD}/clause.o ${BUILD}/parser.o ${BUILD}/heuristic.o
MODULES=${BUILD}/dpll.o ${BUILD}/brute_force.o ${BUILD}/single_thread.o #${BUILD}/cuda.o
HEADERS=${SRC}/list.h ${SRC}/clause.h ${SRC}/parser.h ${SRC}/abrasatcuda.h ${SRC}/interfaces/solve.h ${SRC}/dpll.h ${SRC}/vars.h ${SRC}/consts.h ${SRC}/brute_force.h ${SRC}/interfaces/dispatch.h ${SRC}/heuristic.h

# default dispatching method
DISPATCH_HEADER=${SRC}/single_thread.h
DISPATCH_OBJECT=${BUILD}/single_thread.o

# var containing parameters to be passed to the linker, for creating the final executable
# default : linked against math and ld
LDFLAGS=-lm -ldl
ifeq ($(PARALLEL),pthread)
	LDFLAGS=-lpthread -lm -ldl
	DISPATCH_HEADER=${SRC}/multi_thread.h
	DISPATCH_OBJECT=${BUILD}/multi_thread.o
endif
ifeq ($(PARALLEL),cuda)
	#CC=nvcc
	DISPATCH_OBJECT=${BUILD}/cuda.o
	# TODO
endif

# flags for dynamic libs
DYNFLAGS=-shared

#--------------------------------------------------------------
# targets
#--------------------------------------------------------------

# default target
all: $(TARGETS) $(MODULES)


# launches tests
test: ${DIST}/test_all
	./${DIST}/test_all

main: ${TARGETS}
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
${DIST}/abrasatcuda: ${SRC}/abrasatcuda.c ${BUILD}/parser.o ${BUILD}/clause.o
	$(CC) $(LDFLAGS) $(CFLAGS) $(DBG) $(PROF) ${BUILD}/parser.o ${BUILD}/clause.o ${SRC}/abrasatcuda.c -o ${DIST}/abrasatcuda 

${LIB}/abrasatcuda_dpll_single.so: $(OBJECTS) $(HEADERS) ${BUILD}/dpll.o ${BUILD}/single_thread.o
	$(CC) $(LDFLAGS) $(CFLAGS) $(DBG) $(PROF) $(OBJECTS) -DPARALLEL=single ${BUILD}/dpll.o ${BUILD}/single_thread.o ${SRC}/abrasatcuda.c $(DYNFLAGS) -o ${LIB}/abrasatcuda_dpll_single.so

${LIB}/abrasatcuda_dpll_pthread.so: $(OBJECTS) $(HEADERS) ${BUILD}/dpll.o ${BUILD}/multi_thread.o  
	$(CC) $(LDFLAGS) $(CFLAGS) $(DBG) $(PROF) $(OBJECTS) -DPARALLEL=pthread ${BUILD}/dpll.o ${BUILD}/multi_thread.o  ${SRC}/abrasatcuda.c $(DYNFLAGS) -o ${LIB}/abrasatcuda_dpll_pthread.so

${DIST}/abrasatcuda_cuda: $(OBJECTS) $(HEADERS) $(DISPATCH_OBJECT)
	$(NVCC) $(LDFLAGS) $(NVFLAGS) $(PROF) $(CUDA) $(OBJECTS) $(DISPATCH_OBJECT) ${SRC}/abrasatcuda.c -o abrasatcuda_cuda


# binary for testing
# @deprecated@
${DIST}/test_all: ${SRC}/test.c ${BUILD}/parser.o ${BUILD}/clause.o ${BUILD}/solve.o ${BUILD}/dpll.o
	$(CC) $(CFLAGS) ${SRC}/test.c ${BUILD}/parser.o ${BUILD}/clause.o ${BUILD}/solve.o ${BUILD}/dpll.o -o ${DIST}/test_all


# object files
${BUILD}/parser.o: ${SRC}/parser.c ${SRC}/parser.h
	$(CC) $(CFLAGS) $(DBG) -c ${SRC}/parser.c -o ${BUILD}/parser.o

${BUILD}/clause.o: ${SRC}/clause.c ${SRC}/clause.h
	$(CC) $(CFLAGS) $(DBG) $(PROF) ${SRC}/clause.c -c -o ${BUILD}/clause.o

${BUILD}/dpll.o: ${SRC}/dpll.c ${SRC}/dpll.h ${SRC}/interfaces/solve.h
	$(CC) $(CFLAGS) $(DBG) $(PROF) ${SRC}/dpll.c -c -o ${BUILD}/dpll.o

${BUILD}/brute_force.o: ${SRC}/brute_force.c ${SRC}/brute_force.h ${SRC}/interfaces/solve.h
	$(CC) $(CFLAGS) ${SRC}/brute_force.c -c -o ${BUILD}/brute_force.o

${BUILD}/single_thread.o: ${SRC}/single_thread.c ${SRC}/single_thread.h ${SRC}/interfaces/solve.h
	$(CC) $(CFLAGS) ${SRC}/single_thread.c $(DBG) $(PROF) -c -o ${BUILD}/single_thread.o

${BUILD}/multi_thread.o: ${SRC}/multi_thread.c ${SRC}/multi_thread.h ${SRC}/interfaces/solve.h
	$(CC) $(CFLAGS) -DTHREAD_NUM=${THREAD_NUM} ${SRC}/multi_thread.c $(DBG) $(PROF) -c -o ${BUILD}/multi_thread.o

${BUILD}/heuristic.o: ${SRC}/heuristic.c ${SRC}/heuristic.h
	$(CC) $(CFLAGS) ${SRC}/heuristic.c $(DBG) $(PROF) -c -o ${BUILD}/heuristic.o

${BUILD}/cuda.o: ${SRC}/solve.cu ${SRC}/dpll.c
	$(NVCC) $(CUDA_INCLUDES) $(NVFLAGS) ${SRC}/solve.cu  $(PROF) $(CUDA) -o ${BUILD}/cuda.o



#Cette cible effectue un simple nettoyage des fichiers temporaires qui ont pu être générés
clean:
	@rm -f ${BUILD}/*~ ${BUILD}/a.out ${BUILD}/core
	@rm -f ${BUILD}/*.o
	@rm -f test_all 

#Cette cible effectue un nettoyage complet de tout fichier généré. Elle efface notamment les exécutables.
distclean: clean
	@rm -f $(TARGETS)
