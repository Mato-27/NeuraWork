all: main.out

main.out: obj/main.o obj/matrix.o obj/LinearLayer.o obj/ReLULayer.o obj/SoftmaxLayer.o obj/MSELoss.o obj/Sequential.o obj/SGDOptimizer.o obj/AdamOptimizer.o
	g++ -g -Wall obj/main.o obj/matrix.o obj/LinearLayer.o obj/ReLULayer.o obj/SoftmaxLayer.o obj/MSELoss.o obj/Sequential.o obj/SGDOptimizer.o obj/AdamOptimizer.o -o bin/main.out

obj/main.o: src/main.cpp src/Matrix.h src/layer/LinearLayer.h src/layer/ReLULayer.h src/layer/SoftmaxLayer.h src/loss/MSELoss.h src/Sequential.h
	g++ -g -Wall -c src/main.cpp -o obj/main.o

obj/matrix.o: src/Matrix.h src/Matrix.cpp
	g++ -g -Wall -c src/Matrix.cpp -o obj/matrix.o

obj/LinearLayer.o: src/layer/LinearLayer.h src/layer/LinearLayer.cpp src/layer/Layer.h src/Matrix.h
	g++ -g -Wall -c src/layer/LinearLayer.cpp -o obj/LinearLayer.o

obj/ReLULayer.o: src/layer/ReLULayer.h src/layer/ReLULayer.cpp src/layer/Layer.h src/Matrix.h
	g++ -g -Wall -c src/layer/ReLULayer.cpp -o obj/ReLULayer.o

obj/SoftmaxLayer.o: src/layer/SoftmaxLayer.h src/layer/SoftmaxLayer.cpp src/layer/Layer.h src/Matrix.h
	g++ -g -Wall -c src/layer/SoftmaxLayer.cpp -o obj/SoftmaxLayer.o

obj/MSELoss.o: src/loss/MSELoss.h src/loss/MSELoss.cpp src/loss/Loss.h src/Matrix.h
	g++ -g -Wall -c src/loss/MSELoss.cpp -o obj/MSELoss.o

obj/Sequential.o: src/Sequential.h src/Sequential.cpp src/layer/Layer.h src/Matrix.h
	g++ -g -Wall -c src/Sequential.cpp -o obj/Sequential.o

obj/SGDOptimizer.o: src/optimizer/SGDOptimizer.h src/optimizer/SGDOptimizer.cpp src/optimizer/Optimizer.h src/Matrix.h
	g++ -g -Wall -c src/optimizer/SGDOptimizer.cpp -o obj/SGDOptimizer.o

obj/AdamOptimizer.o: src/optimizer/AdamOptimizer.h src/optimizer/AdamOptimizer.cpp src/optimizer/Optimizer.h src/Matrix.h
	g++ -g -Wall -c src/optimizer/AdamOptimizer.cpp -o obj/AdamOptimizer.o

docu:
	doxygen doc/doxyfile

clean: 
	rm -rf obj/* doc/html/*

veryclean: clean
	rm -rf bin/*