#include <ppso/particle_swarm.hpp>

#include <iostream>

// f(x) = ||x||^2
// note: can handle constraints by returning
//       "infinity" (max double value) when infeasible
class objective_function {
public:
  double operator()(const std::vector<double> &x) {
    double ret = 0;
    for (double i: x) {
      ret += (i * i);
    }
    return ret;
  }
};

int main(int argc, char *argv[])
{
  std::vector<double> min_p = {-1000, -2000};
  std::vector<double> max_p = {1000, 4000};
  std::vector<double> wts(2000, 0.6); // each of 2000 iterations have weight 0.6

  // perform ppso
  auto ret = ppso::Particle_Swarm{
    objective_function(),
    2, // number of dimensions
    min_p, // vector of minimum values for each dimension
    max_p, // vector of maximum values for each dimension
    std::max(std::thread::hardware_concurrency(), (unsigned int)2), // number of groups (threads)
    20, // number of particles
    2000.0, // maximum velocity
    wts, // vector of weights for each iteration
    2, // personal influence constant
    2, // group influence constant
    4, // every 4 iterations, communication strategy 1
    8, // every 8 iterations, communication strategy 2
    2000 // 2000 iterations
  }.run();

  // print x*
  std::cout << std::fixed << "x* = [ ";
  for (double x: ret.first) {
    std::cout << x << " ";
  }
  std::cout << "]'\n";

  // print f(x*)
  std::cout << "f(x*) = " << ret.second << std::endl;
  
  return 0;
}

