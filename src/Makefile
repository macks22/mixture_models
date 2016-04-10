
all: mixture/ccomp.pyx mixture/cdist.pyx
	python setup.py build_ext --inplace
	mv *.so mixture

clean:
	rm -rf build
	cd mixture && make clean && cd ..


.PHONY: clean
