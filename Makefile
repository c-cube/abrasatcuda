#Variable contenant le nom du compilateur
CC=gcc
#Variable contenant les options passées au compilateur
CFLAGS=-Wall -Werror -std=c99 -pedantic -Os -g -c # -m32
#L'option -Wall affiche tous les messages d'alertes (warnings)
#L'option -Werror traite une simple alerte comme une erreur (stoppant ainsi lq compilation)
#L'option -std= permet de fixer la norme ISO du C que le compilateur va utiliser pour vérifier la validité du programme. 
#L'option -pedantic empeche l'utilisation des extensions et impose de se conformer à la version stricte de la norme ISO
#L'option -O permet de définir le degré d'optimisation appliqué par le compilateur (0 est la valeur par défaut, aucune optimisation)
#L'option -g compile le programme en laissant les informations nécessaires pour l'utilisation d'un debugger lors de l'exécution.
#L'option -m indique que le code généré doit être pour un environement 32 bits
#L'option -c indique que le compilateur doit se contenter de la génération d'un fichier objet. La création d'un exécutable se fera si nécessaire dans un second temps.

#Variable contenant les options passées au compilateur pour l'édition de liens
LDFLAGS=

#Variable contenant la liste des cibles 
TARGETS=abrasatcuda

#all est la cible par défaut. 
#on la fait correspondre à l'ensemble des cibles qu'on souhaite exécuter
all: $(TARGETS)

#Cette cible effectue la compilation de notre commande.
#Elle n'est exécutée que si le fichier mon_cp.c est plus recent que le fichier exécutable mon_cp
abrasatcuda: clause.o abrasatcuda.o
	#effectue la compilation et linke.
	$(CC) $(LDFLAGS) clause.o abrasatcuda.o -o abrasatcuda

clause.o: clause.c
	$(CC) $(CFLAGS) clause.c 

abrasatcuda.o: abrasatcuda.c
	$(CC) $(CFLAGS) abrasatcuda.c

#Cette cible effectue un simple nettoyage des fichiers temporaires qui ont pu être générés
clean:
	rm -f *~ a.out core
	rm -f *.o

#Cette cible effectue un nettoyage complet de tout fichier généré. Elle efface notamment les exécutables.
distclean: clean
	rm -f $(TARGETS)
