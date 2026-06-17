all: main.out

main.out: obj/main.o obj/matrix.o obj/layer.o obj/MSELoss.o obj/Sequential.o obj/SGDOptimizer.o obj/AdamOptimizer.o
	g++ -g -Wall obj/main.o obj/matrix.o  obj/layer.o obj/MSELoss.o obj/Sequential.o obj/SGDOptimizer.o obj/AdamOptimizer.o -o bin/main.out

obj/main.o: src/main.cpp src/Matrix.h src/Layer.h
	g++ -g -Wall -c src/main.cpp -o obj/main.o

obj/matrix.o: src/Matrix.h src/Matrix.cpp
	g++ -g -Wall -c src/Matrix.cpp -o obj/matrix.o

obj/layer.o: src/Layer.h src/Layer.cpp
	g++ -g -Wall -c src/Layer.cpp -o obj/layer.o

obj/MSELoss.o: src/MSELoss.h src/MSELoss.cpp
	g++ -g -Wall -c src/MSELoss.cpp -o obj/MSELoss.o

obj/Sequential.o: src/Sequential.h src/Sequential.cpp
	g++ -g -Wall -c src/Sequential.cpp -o obj/Sequential.o

obj/SGDOptimizer.o: src/SGDOptimizer.h src/SGDOptimizer.cpp
	g++ -g -Wall -c src/SGDOptimizer.cpp -o obj/SGDOptimizer.o

obj/AdamOptimizer.o: src/AdamOptimizer.h src/AdamOptimizer.cpp
	g++ -g -Wall -c src/AdamOptimizer.cpp -o obj/AdamOptimizer.o

docu:
	doxygen doc/doxyfile

clean: 
	rm -rf obj/* doc/html/*

veryclean: clean
	rm -rf bin/*