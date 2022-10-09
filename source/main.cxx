// N Pendulum simulation

#include <stdio.h>
#include <cmath>
#include <types/type_alias.hxx>
#include <graphics/raylib_window/raylib_window.hxx>

#include <eigen3/Eigen/Dense>

template<u64 N>
using Vecf64 = Eigen::Vector<f64, N>;

template<u64 N, u64 M>
using Matf64 = Eigen::Matrix<f64, N, M>;

struct pos {
  f64 x, y;
};

template<u64 N>
class Pendulum {
public:
  Pendulum(f64 gravity = 9.8) : g(gravity) {
    thetas.fill(0.5 * M_PI);
    theta_dots.fill(0);
  }

  void step(f64 dt) {
    RK4(dt);
  }

  vec<pos> get_coords() {
    f64 x = 0, y = 0;
    vec<pos> coords;
    for (u64 i = 0; i < N; ++i) {
      auto theta = thetas.coeff(i);
      x += sin(theta);
      y += cos(theta);
      coords.push_back({x, y});
    }
    return coords;
  }

private:
  Matf64<N, N> gen_a(Vecf64<N> angs) {
    Matf64<N, N> a;
    for (i64 i = 0; i < N; ++i) {
      for (i64 j = 0; j < N; ++j) {
        // TODO: This might need to be j, i instead
        a.coeffRef(i, j) = (N - std::max(i, j)) * cos(angs.coeff(i) - angs.coeff(j));
      }
    }
    return a;
  }

  Vecf64<N> gen_b(Vecf64<N> angs, Vecf64<N> dangs) {
    Vecf64<N> b;
    for (i64 i = 0; i < N; ++i) {
      f64 b_i = 0;
      for (i64 j = 0; j < N; ++j) {
        b_i -= (N - std::max(i, j)) * sin(angs.coeff(i) - angs.coeff(j)) * dangs.coeff(j) * dangs.coeff(j);
      }
      b_i -= g * (N - i) * sin(angs.coeff(i));
      b.coeffRef(i) = b_i;
    }
    return b;
  }

  std::pair<Vecf64<N>, Vecf64<N>> f(Vecf64<N> angs, Vecf64<N> dangs) {
    auto a = gen_a(angs);
    auto b = gen_b(angs, dangs);
    auto x = a.colPivHouseholderQr().solve(b);
    return std::make_pair(dangs, x);
  }

  void RK4(f64 dt) {
    auto k1 = f(thetas, theta_dots);
    auto k2 = f(thetas + k1.first * 0.5 * dt, theta_dots + k1.second * 0.5 * dt);
    auto k3 = f(thetas + k2.first * 0.5 * dt, theta_dots + k2.second * 0.5 * dt);
    auto k4 = f(thetas + k3.first * dt, theta_dots + k3.second * dt);

    auto delta_thetas = (k1.first + k2.first * 2 + k3.first * 2 + k4.first) * dt / 6.0;
    auto delta_theta_dots = (k1.second + k2.second * 2 + k3.second * 2 + k4.second) * dt / 6.0;
    thetas += delta_thetas;
    theta_dots += delta_theta_dots;
  }

  f64 g;
  Vecf64<N> thetas;
  Vecf64<N> theta_dots;
};

int main(int, char**) {
  printf("Hello, world!");
  Pendulum<4> p{};
  for(;;) {
    p.step(1/60.0);
  }
  return 0;
}
