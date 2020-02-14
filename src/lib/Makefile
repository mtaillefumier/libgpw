CXXFLAGS   = -DNDEBUG -DHAVE_MKL -g2 -O2 -Wall -Wextra -mavx2 -I/opt/intel/mkl/intel64/include -ftree-loop-vectorize -ftree-vectorize

all: grid_base_ref_miniapp.x collocate

grid_base_ref_replay.o: grid_base_ref_c.o

includes = $(wildcard *.hpp)

%.o: %.cc $(includes)
	$(CXX) -c $(CXXFLAGS) $<
%.o: %.cpp $(includes)
	$(CXX) -c $(CXXFLAGS) $<

grid_base_ref_miniapp.x: rt_graph.o grid_base_ref_miniapp.o grid_base_ref_replay.o grid_base_ref_c.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lm

collocate : collocate.o rt_graph.o
	$(CXX) $(CXXFLAGS) -lmkl_rt -o $@ $^ -lm
clean:
	rm -fv grid_base_ref*.o grid_base_ref*.x collocate.o collocate

#EOF
