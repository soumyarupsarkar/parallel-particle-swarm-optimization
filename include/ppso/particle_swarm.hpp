#ifndef _particle_swarm_hpp
#define _particle_swarm_hpp

#include <algorithm>
#include <functional>
#include <limits>
#include <random>
#include <utility>
#include <thread>
#include <vector>

namespace ppso {
  class Particle_Swarm {
    // individual particle properties
    std::vector<std::vector<std::vector<double> > > m_pos; // X_{i,j}
    std::vector<std::vector<double > > m_f_pos;

    std::vector<std::vector<std::vector<double> > > m_best; // P_{i,j}
    std::vector<std::vector<double > > m_f_best;

    std::vector<std::vector<std::vector<double> > > m_velocity; // V_{i,j}
    
    // group properties
    std::vector<std::vector<double> > m_g_best; // G_{j}
    std::vector<double> m_f_g_best;

    // swarm property
    std::vector<double> m_s_best;
    double m_f_s_best;

    // problem description
    const std::function<double(const std::vector<double> &)> m_obj_func;
    unsigned int m_dim; // dimension of input vectors
    
    const unsigned int m_num_groups;
    const unsigned int m_num_particles;

    const double m_v_max;
    const std::vector<double> &m_weights; // W_{t}
    const unsigned int m_constant_1; // C_1
    const unsigned int m_constant_2; // C_2

    // iteration count and constants
    const unsigned int m_loose_corr_strategy; // R_1
    const unsigned int m_strong_corr_strategy; // R_2
    const unsigned int m_max_iter;
    unsigned int m_iter; // t

    // random number generator for r_{1}, r_{2}
    std::default_random_engine m_randgen;
    std::uniform_real_distribution<double> m_urdist; // [0, 1]

    // helper functions
    void for_each_group(int i);
    void loose_corr_strategy_for_group(unsigned int i);
    void strong_corr_strategy_for_group();

  public:
    // min_init and max_init are lower and upper bounds of the initial search space
    Particle_Swarm(const std::function<double(const std::vector<double> &)> obj_func,
                   const unsigned int n_dimensions,
                   const std::vector<double> &min_init,
                   const std::vector<double> &max_init,
                   const unsigned int num_groups,
                   const unsigned int num_particles,
                   const double max_velocity,
                   const std::vector<double> &weights,
                   const unsigned int constant_1,
                   const unsigned int constant_2,
                   const unsigned int loose_corr_strategy,
                   const unsigned int strong_corr_strategy,
                   const unsigned int max_iter);

    ~Particle_Swarm();

    std::pair<std::vector<double>, double> run();
  };
}

#endif


