all: main.out

main.out: obj/main.o obj/matrix.o obj/layer.o
	g++ -g obj/main.o obj/matrix.o  obj/layer.o -o bin/main.out

obj/main.o: src/main.cpp src/Matrix.h src/Layer.h
	g++ -g -Wall -c src/main.cpp -o obj/main.o

obj/matrix.o: src/Matrix.h src/Matrix.cpp
	g++ -g -Wall -c src/Matrix.cpp -o obj/matrix.o

obj/layer.o: src/Layer.h src/Layer.cpp
	g++ -g -Wall -c src/Layer.cpp -o obj/layer.o

docu:
	doxygen doc/doxyfile

clean: 
	rm -rf obj/* doc/html/*

veryclean: clean
	rm -rf bin/*