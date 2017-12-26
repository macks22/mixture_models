DIST=/Users/msweeney/anaconda2
PYHOME=$(DIST)/include/python2.7/
NUMPY=$(DIST)/pkgs/numpy-1.10.4-py27_0/lib/python2.7/site-packages/numpy/core/include/
CFLAGS=-fPIC -ffast-math -O3 -I$(PYHOME) -I$(NUMPY) -l$(PYHOME)

# Note: numpy include path can be gotten using:
# 	`python -c "import numpy; numpy.get_include()"`

CDIST=cdist
CCOMP=ccomp

all: cdist ccomp

cdist: $(CDIST).pyx
	cython -a $(CDIST).pyx
	gcc -c $(CFLAGS) $(CDIST).c
	gcc -shared $(CDIST).o -o $(CDIST).so


ccomp: $(CCOMP).pyx
	cython -a $(CCOMP).pyx
	gcc -c $(CFLAGS) $(CCOMP).c
	gcc -shared $(CCOMP).o -o $(CCOMP).so

clean:
	rm -f $(CDIST).c
	rm -f $(CDIST).o
	rm -f $(CDIST).html
	rm -f $(CCOMP).c
	rm -f $(CCOMP).o
	rm -f $(CCOMP).html


PHONY: clean
