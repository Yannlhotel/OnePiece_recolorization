# Détection du système d'exploitation
ifeq ($(OS),Windows_NT)
    TARGET_MAKEFILE := Makefile.win
else
    TARGET_MAKEFILE := Makefile.linux
endif

# Cette règle "attrape-tout" (%) redirige n'importe quelle commande (ex: make setup)
# vers le fichier cible (ex: make -f Makefile.win setup)
%:
	@make -f $(TARGET_MAKEFILE) $@

# Gestion du cas où on lance juste "make" sans argument (par défaut : help ou setup)
default:
	@make -f $(TARGET_MAKEFILE)