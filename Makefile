#Variable contenant le nom du compilateur
CC=gcc
#Variable contenant les options passées au compilateur
CFLAGS=-Wall -pedantic -Os -g  -std=gnu99 #-m32 -Werror
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
TARGETS=abrasatcuda test_all
OBJECTS=${BUILD}/abrasatcuda.o ${BUILD}/clause.o ${BUILD}/parser.o ${BUILD}/solve.o
HEADERS=${SRC}/list.h ${SRC}/clause.h ${SRC}/parser.h ${SRC}/abrasatcuda.h ${SRC}/solve.h ${SRC}/dpll.h ${SRC}/vars.h


# dossiers divers
SRC=src
BUILD=build



# default target
all: $(TARGETS)

# launches tests
test: test_all
	./test_all

main: abrasatcuda
	echo "example.cnf :"
	./abrasatcuda tests/example.cnf
	echo "trivial.cnf :"
	./abrasatcuda tests/trivial.cnf
	echo "false.cnf :"
	./abrasatcuda tests/false.cnf

check: check.hs
	ghc -O2 --make check.hs -o check

count:
	grep -v '^[ ]*$$' ./${SRC}/{*.h,*.c} | wc -l	
# This targets compiles the main binary
abrasatcuda:  $(OBJECTS)
	$(CC) $(LDFLAGS)  $(OBJECTS) -o abrasatcuda

# binary for testing
test_all: ${SRC}/test.c ${BUILD}/parser.o ${BUILD}/clause.o ${BUILD}/solve.o
	$(CC) $(CFLAGS) ${SRC}/test.c ${BUILD}/parser.o ${BUILD}/clause.o ${BUILD}/solve.o -o test_all


# object files

${BUILD}/abrasatcuda.o: ${SRC}/abrasatcuda.c $(HEADERS)
	$(CC) $(CFLAGS) -c ${SRC}/abrasatcuda.c -o ${BUILD}/abrasatcuda.o

${BUILD}/parser.o: ${SRC}/parser.c $(HEADERS)
	$(CC) $(CFLAGS) -c ${SRC}/parser.c -o ${BUILD}/parser.o

${BUILD}/clause.o: ${SRC}/clause.c ${SRC}/clause.h
	$(CC) $(CFLAGS) ${SRC}/clause.c -c -o ${BUILD}/clause.o

${BUILD}/solve.o: ${SRC}/solve.c ${SRC}/solve.h
	$(CC) $(CFLAGS) ${SRC}/solve.c -c -o ${BUILD}/solve.o




#Cette cible effectue un simple nettoyage des fichiers temporaires qui ont pu être générés
clean:
	rm -f ${BUILD}/*~ ${BUILD}/a.out ${BUILD}/core
	rm -f ${BUILD}/*.o
	rm -f test_all abrasatcuda

#Cette cible effectue un nettoyage complet de tout fichier généré. Elle efface notamment les exécutables.
distclean: clean
	rm -f $(TARGETS)
