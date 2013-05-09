#include <ppso/particle_swarm.hpp>

#include <cmath>
#include <iostream>

class Rosenbrock_function {
public:
  double operator()(const std::vector<double> &x) {
    double ret = 0;
    for (int i = 0; i < x.size() - 1; ++i) {
      ret += std::pow(x[i] - 1.0, 2) + \
        (100.0 * std::pow(x[i + 1] - (x[i] * x[i]), 2));
    }
    return ret;
  }
};

int main(int argc, char *argv[])
{
  std::vector<double> min_p(16, -4.0);
  std::vector<double> max_p(16, 4.0);
  std::vector<double> wts(20000, 0.6); // each of 20000 iterations have weight 0.6

  // perform ppso
  auto ret = ppso::Particle_Swarm{
    Rosenbrock_function(),
    16, // number of dimensions
    min_p, // vector of minimum values for each dimension
    max_p, // vector of maximum values for each dimension
    std::thread::hardware_concurrency(), // number of groups (threads)
    40, // number of particles
    0.004, // maximum velocity
    wts, // vector of weights for each iteration
    2, // personal influence constant
    2, // group influence constant

    // NOT IMPLEMENTED:
    4, // every 4 iterations, communication strategy 1
    8, // every 8 iterations, communication strategy 2

    20000 // 20000 iterations
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

