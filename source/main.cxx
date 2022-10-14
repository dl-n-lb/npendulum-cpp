// N Pendulum simulation

#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <types/type_alias.hxx>
#define RAYLIB_WINDOW_IMPL
#define RAYLIB_RENDERER_IMPL
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
  Pendulum(f64 scale = 1.0, f64 gravity = 9.8) : s(scale), g(gravity) {
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
      x += sin(theta)*s;
      y += cos(theta)*s;
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
    Vecf64<N> x = a.colPivHouseholderQr().solve(b);
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
  f64 s;
  Vecf64<N> thetas;
  Vecf64<N> theta_dots;
};


static constexpr u32 scr_width = 1600;
static constexpr u32 scr_height = 1000;
static constexpr u64 fps = 240;

// TODO: CUSTOM OPTIONS FOR WTV LIKE ACCURACY, ETC (N IS COMPILE TIME ATM)
// PERHAPS THERE CAN BE A DYNAMIC VERSION
// THERE IS NO WAY THAT HIGHER NUMBERS ARE ACCURATE RIGHT?
// PERHAPS PRINT OUT ENERGY OR LAGRANGIAN
int main(int, char**) {
  printf("Hello, world!");
  Pendulum<8> p{40};

  RaylibWindow rw = RaylibWindow{scr_width, scr_height, "Gamer"};

  auto start = std::chrono::system_clock::now();

  u64 frame_cnt = 0;

  rw.setTargetFPS(fps);

  auto update_fn = [&p](RaylibWindow* win) {
    p.step(1.0/fps);
    win->getRenderer(BLACK);
    pos prev = {0, 0};
    for(auto coord : p.get_coords()) {
      // TODO: Narrowing here needs to be explicit
      DrawLine(prev.x + scr_width/2, prev.y + scr_height/2,
               coord.x + scr_width/2, coord.y + scr_height/2, RAYWHITE);
      prev = coord;
    }
  };
  while(!rw.shouldClose()) {
    ++frame_cnt;
    rw.Update(update_fn);
  }

  auto dur = std::chrono::duration_cast<std::chrono::seconds>
    (std::chrono::system_clock::now() - start).count();

  printf("Average FPS: %lf\n", (f64)frame_cnt / dur);
  return 0;
}
