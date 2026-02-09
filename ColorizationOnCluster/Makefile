# Operating system detection
ifeq ($(OS),Windows_NT)
    TARGET_MAKEFILE := Makefile.win
else
    TARGET_MAKEFILE := Makefile.linux
endif

# This catch-all rule (%) redirects any command (e.g., make setup)
# to the target file (e.g., make -f Makefile.win setup)
%:
	@make -f $(TARGET_MAKEFILE) $@

# Handle the case where "make" is called without arguments (default: help or setup)
default:
	@make -f $(TARGET_MAKEFILE)