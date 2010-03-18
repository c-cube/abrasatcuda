# switch debug on/off
DEBUG=n # yes,no,prod
PROFILE=yes

#Variable contenant le nom du compilateur
CC=gcc

CFLAGS=-Wall -pedantic -Os -std=gnu99
#Variable contenant les options passées au compilateur
DBG=
ifeq ($(DEBUG),yes)
	DBG=-g -DDEBUG=1 #-m32 -Werror
endif
ifeq ($(DEBUG),prod)
	DBG=-DNDEBUG=1 # NDEBUG disables all assert() statements
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

export LD_LIBRARY_PATH := .
#Variable contenant les options passées au compilateur pour l'édition de liens
LDFLAGS=

#Variable contenant la liste des cibles 
TARGETS=abrasatcuda_bf abrasatcuda_dpll 
OBJECTS=${BUILD}/clause.o ${BUILD}/parser.o  
MODULES=${BUILD}/dpll.o ${BUILD}/brute_force.o ${BUILD}/single_thread.o
HEADERS=${SRC}/list.h ${SRC}/clause.h ${SRC}/parser.h ${SRC}/abrasatcuda.h ${SRC}/interfaces/solve.h ${SRC}/dpll.h ${SRC}/vars.h ${SRC}/consts.h ${SRC}/brute_force.h ${SRC}/single_thread.h ${SRC}/interfaces/dispatch.h Makefile


# dossiers divers
SRC=src
BUILD=build



# default target
all: $(TARGETS) $(MODULES)


# launches tests
test: test_all
	./test_all

main: abrasatcuda_bf abrasatcuda_dpll
	@echo -e "\n\e[45;4mexample.cnf :\e[m"
	@./abrasatcuda tests/example.cnf
	@echo -e "\n\e[45;4mtrivial.cnf :\e[m"
	@./abrasatcuda tests/trivial.cnf
	@echo -e "\n\e[45;4mfalse.cnf :\e[m"
	@./abrasatcuda tests/false.cnf
	@echo -e "\n\e[45;4mquinn.cnf :\e[m"
	@./abrasatcuda tests/quinn.cnf


prof:
	gprof ./abrasatcuda


check: check.hs
	ghc -O2 --make check.hs -o check

count:
	@echo "number of code/comment lines : "; grep -v '^[ ]*$$' ./${SRC}/{*.h,*.c} | wc -l	


# This targets compiles the main binary
abrasatcuda_bf: $(OBJECTS) $(HEADERS) ${BUILD}/brute_force.o  ${BUILD}/single_thread.o
	$(CC) $(LDFLAGS) $(CFLAGS) $(OBJECTS) ${BUILD}/brute_force.o ${BUILD}/single_thread.o ${SRC}/abrasatcuda.c -o abrasatcuda_bf


abrasatcuda_dpll: $(OBJECTS) $(HEADERS) ${BUILD}/dpll.o ${BUILD}/single_thread.o 
	$(CC) $(LDFLAGS) $(CFLAGS) $(DBG) $(PROF) $(OBJECTS) ${BUILD}/dpll.o ${BUILD}/single_thread.o ${SRC}/abrasatcuda.c -o abrasatcuda_dpll



# binary for testing
# @deprecated@
test_all: ${SRC}/test.c ${BUILD}/parser.o ${BUILD}/clause.o ${BUILD}/solve.o ${BUILD}/dpll.o
	$(CC) $(CFLAGS) ${SRC}/test.c ${BUILD}/parser.o ${BUILD}/clause.o ${BUILD}/solve.o ${BUILD}/dpll.o -o test_all


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


#Cette cible effectue un simple nettoyage des fichiers temporaires qui ont pu être générés
clean:
	@rm -f ${BUILD}/*~ ${BUILD}/a.out ${BUILD}/core
	@rm -f ${BUILD}/*.o
	@rm -f test_all abrasatcuda_bf abrasatcuda_dpll

#Cette cible effectue un nettoyage complet de tout fichier généré. Elle efface notamment les exécutables.
distclean: clean
	@rm -f $(TARGETS)
