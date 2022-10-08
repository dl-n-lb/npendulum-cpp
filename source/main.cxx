// N Pendulum simulation

#include <stdio.h>
#include <cmath>
#include <types/type_alias.hxx>

#include <eigen3/Eigen/Dense>

struct pos {
  f32 x, y;
};

template<u64 N>
class Pendulum {
public:
  Pendulum(f64 gravity = 9.8) : g(gravity) {
    thetas.fill(0.5 * M_PI);
    theta_dots.fill(0);
  }

  void init() {
    gen_A();
    gen_b();
  }

  void step();

  vec<pos> get_coords();

private:
  void gen_A() {
    for (i64 i = 0; i < N; ++i) {
      for (i64 j = 0; j < N; ++j) {
        // NOTE: This might need to be j, i instead
        A.coeffRef(i, j) = (N - std::max(i, j)) * cos(thetas[i] - thetas[j]);
      }
    }
  }

  void gen_b() {
    for (i64 i = 0; i < N; ++i) {
      f64 b_i = 0;
      for (i64 j = 0; j < N; ++j) {
        b_i -= (N - std::max(i, j)) * sin(thetas[i] - thetas[j]) * theta_dots[j] * theta_dots[j];
      }
      b_i -= g * (N - i) * sin(thetas[i]);
      b.coeffRef(i) = b_i;
    }
  }

  void f();

  void RK4();

  f64 g;
  arr<N, f64> thetas;
  arr<N, f64> theta_dots;

  Eigen::Matrix<f64, N, N> A;
  Eigen::Vector<f64, N> b;
};

int main(int, char**) {
  printf("Hello, world!");
  Pendulum<4> p{};
  return 0;
}
