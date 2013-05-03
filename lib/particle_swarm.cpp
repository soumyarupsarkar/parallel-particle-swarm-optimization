#include <ppso/particle_swarm.hpp>


namespace {
  std::vector<std::thread> workers;
}

namespace ppso {
  Particle_Swarm::Particle_Swarm(const std::function<double(const std::vector<double> &)> obj_func,
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
                                 const unsigned int max_iter)
  : m_obj_func(obj_func),
    m_dim(n_dimensions),
    m_weights(weights),
    m_v_max(max_velocity),
    m_constant_1(constant_1),
    m_constant_2(constant_2),
    m_loose_corr_strategy(loose_corr_strategy),
    m_strong_corr_strategy(strong_corr_strategy),
    m_max_iter(max_iter),
    m_num_groups(num_groups),
    m_num_particles(num_particles),
    m_iter(0)
  {
    // initialize uniform distributions for initialization
    std::default_random_engine randgen;
    std::vector<std::uniform_real_distribution<double> > urdist;
    urdist.resize(m_dim);
    for (int i = 0; i < m_dim; ++i) {
      urdist[i] = std::uniform_real_distribution<double>(min_init[i], max_init[i]);
    }

    // initialize uniformly-distributed positions, best positions, velocities, and function values
    m_pos.resize(num_groups);
    m_velocity.resize(num_groups);
    m_best.resize(num_groups);
    m_f_pos.resize(num_groups);
    m_f_best.resize(num_groups);

    for (int i = 0; i < num_groups; ++i) { // for each group
      m_pos[i].resize(num_particles);
      m_velocity[i].resize(num_particles);
      m_best[i].resize(num_particles);
      m_f_pos[i].resize(num_particles);
      m_f_best[i].resize(num_particles);

      for (int j = 0; j < num_particles; ++j) { // for each particle
        m_pos[i][j].resize(m_dim);
        m_velocity[i][j].resize(m_dim);
        m_best[i][j].resize(m_dim);

        for (int k = 0; k < m_dim; ++k) {
          m_velocity[i][j][k] = m_best[i][j][k] = m_pos[i][j][k] = urdist[k](randgen);
        }
        m_f_best[i][j] = m_f_pos[i][j] = obj_func(m_pos[i][j]);
      }
    }

    // find and set bests (with function values) for each group
    std::vector<double>::iterator curr_best;
    unsigned int curr_best_index;
    m_g_best.resize(num_groups);
    m_f_g_best.resize(num_groups);
    for (int i = 0; i < num_groups; ++i) {
      m_g_best[i].resize(m_dim);

      curr_best = std::min_element(begin(m_f_best[i]), end(m_f_best[i]));
      m_f_g_best[i] = *curr_best;

      curr_best_index = curr_best - begin(m_f_best[i]);
      m_g_best[i] = m_best[i][curr_best_index];
    }

    // set swarm function value to maximum possible
    m_f_s_best = std::numeric_limits<double>::max();
    m_s_best.push_back(0); 
  }

  Particle_Swarm::~Particle_Swarm() {
    // nothing to do.
  }

  void Particle_Swarm::for_each_group(int i) {
    for (int j = 0; j < m_num_particles; ++j) { // for each particle
      m_f_pos[i][j] = m_obj_func(m_pos[i][j]); // evaluate function
      if (m_f_pos[i][j] < m_f_best[i][j]) { // set personal bests, if need be
        m_f_best[i][j] = m_f_pos[i][j];
        m_best[i][j] = m_pos[i][j];
      }

      // set velocity and position
      double r1 = m_urdist(m_randgen);
      double r2 = m_urdist(m_randgen);

      for (int k = 0; k < m_dim; ++k) {
        auto candidate_velocity = m_weights[m_iter] * m_velocity[i][j][k] + \
          m_constant_1 * r1 * (m_best[i][j][k] - m_pos[i][j][k]) + \
          m_constant_2 * r2 * (m_g_best[i][k] - m_pos[i][j][k]);

        m_velocity[i][j][k] = std::min(candidate_velocity, m_v_max);

        m_pos[i][j][k] += m_velocity[i][j][k];
      }
    }

    // set group properties, if need be
    auto group_min = std::min_element(begin(m_f_best[i]), end(m_f_best[i]));
    unsigned int group_min_index = group_min - begin(m_f_best[i]);
    if (*group_min < m_f_g_best[i]) {
      m_f_g_best[i] = *group_min;
      m_g_best[i] = m_best[i][group_min_index];
    }
  }

  void Particle_Swarm::loose_corr_strategy_for_group(unsigned int i) {
    // not yet implemented
  }

  void Particle_Swarm::strong_corr_strategy_for_group() {
    // not yet implemented
  }


  std::pair<std::vector<double>, double> Particle_Swarm::run() {
    std::vector<double>::iterator poss_swarm_best;
    while (++m_iter < m_max_iter) {
      for (int i = 0; i < m_num_groups; ++i) {
        workers.push_back(std::thread{&Particle_Swarm::for_each_group, this, i});
      }
      for (std::thread &t: workers) {
        if (t.joinable()) {
          t.join(); 
        }
      }
      
      // calculate swarm best
      poss_swarm_best = std::min_element(begin(m_f_g_best), end(m_f_g_best));
      unsigned int poss_swarm_best_index = poss_swarm_best - begin(m_f_g_best);
      if (*poss_swarm_best < m_f_s_best) {
        m_f_s_best = *poss_swarm_best;
        m_s_best = m_g_best[poss_swarm_best_index];
      }

      // communication
      if (((m_iter + 1) % m_loose_corr_strategy) == 0) {
        for (int i = 0; i < m_num_groups; ++i) {
          workers.push_back(std::thread{                            \
              &Particle_Swarm::loose_corr_strategy_for_group, this,
                poss_swarm_best_index});
        }
        for (std::thread &t: workers) {
          if (t.joinable()) {
            t.join(); 
          }
        }
      }

      if (((m_iter + 1) % m_strong_corr_strategy) == 0) {
        for (int i = 0; i < m_num_groups; ++i) {
          workers.push_back(std::thread{                                \
              &Particle_Swarm::strong_corr_strategy_for_group, this});
        }
        for (std::thread &t: workers) {
          if (t.joinable()) {
            t.join(); 
          }
        }
      }
    }
    return std::make_pair(m_s_best, m_f_s_best);
  }
}


