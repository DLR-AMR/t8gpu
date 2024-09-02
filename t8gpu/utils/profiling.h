#ifndef UTILS_PROFILING_H
#define UTILS_PROFILING_H

#include <cstdio>
#include <ctime>

#define T8GPU_TIME(expr)                                                                                       \
  do {                                                                                                         \
    std::timespec ts_begin{}, ts_end{};                                                                        \
    std::timespec_get(&ts_begin, TIME_UTC);                                                                    \
    (expr);                                                                                                    \
    std::timespec_get(&ts_end, TIME_UTC);                                                                      \
    double time_spent = static_cast<double>(ts_end.tv_sec - ts_begin.tv_sec) +                                 \
                        static_cast<double>(ts_end.tv_nsec - ts_begin.tv_nsec) * 1e-9;                         \
    std::fprintf(stderr, "%20.20s:%5d       %-40.40s %.5e sec \n", __FUNCTION__, __LINE__, #expr, time_spent); \
  } while (0)

#define T8GPU_TIMER_START(name)                     \
  int           ln_begin_##name = __LINE__;         \
  std::timespec ts_begin_##name{}, ts_end_##name{}; \
  std::timespec_get(&ts_begin_##name, TIME_UTC)

#define T8GPU_TIMER_STOP(name)                                                                       \
  do {                                                                                               \
    int ln_end_##name = __LINE__;                                                                    \
    std::timespec_get(&ts_end_##name, TIME_UTC);                                                     \
    double time_spent = static_cast<double>(ts_end_##name.tv_sec - ts_begin_##name.tv_sec) +         \
                        static_cast<double>(ts_end_##name.tv_nsec - ts_begin_##name.tv_nsec) * 1e-9; \
    std::fprintf(stderr,                                                                             \
                 "%20.20s:%5d-%-5d %-40.40s %.5e sec \n",                                            \
                 __FUNCTION__,                                                                       \
                 ln_begin_##name,                                                                    \
                 ln_end_##name,                                                                      \
                 #name,                                                                              \
                 time_spent);                                                                        \
  } while (0)

#endif  // UTILS_PROFILING_H
