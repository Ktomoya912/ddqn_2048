#pragma once
#include <assert.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include "play_table.h"
using namespace std;

struct state_t {
  int board[9];
  int score;

  void print() const {
    for (int j = 0; j < 3; j++) {
      for (int i = 0; i < 3; i++) {
        printf("%3d", board[j * 3 + i]);
      }
      printf("\n");
    }
    printf("score = %d\n", score);
  }
  state_t clone() const {
    state_t ret;
    for (int i = 0; i < 9; i++) ret.board[i] = board[i];
    ret.score = score;
    return ret;
  }
};

inline void putNewTile(state_t *state) {
  int empty[9];
  int count = 0;
  for (int i = 0; i < 9; i++) {
    if (state->board[i] == 0) {
      empty[count++] = i;
    }
  }
  if (count != 0) {
    int pos = empty[rand() % count];
    state->board[pos] = ((double)rand() / RAND_MAX) < 0.9 ? 1 : 2;
  }
}

inline void putNewTile(state_t *state, mt19937 &mt) {
  int empty[9];
  int count = 0;
  for (int i = 0; i < 9; i++) {
    if (state->board[i] == 0) {
      empty[count++] = i;
    }
  }
  if (count != 0) {
    int pos = empty[uniform_int_distribution<>(0, count - 1)(mt)];
    state->board[pos] =
        (uniform_real_distribution<>(0.0, 1.0)(mt) < 0.9) ? 1 : 2;
  }
}

inline state_t initGame() {
  state_t state;
  memset(&state.board, 0, sizeof(int) * 9);
  state.score = 0;
  putNewTile(&state);
  putNewTile(&state);
  return state;
}

inline bool play_up(const state_t &org, state_t *dest) {
  bool moved = false;
  int score = org.score;
  const play_table &ta036 = pde[org.board[6]][org.board[3]][org.board[0]];
  dest->board[0] = ta036.p0;
  dest->board[3] = ta036.p1;
  dest->board[6] = ta036.p2;
  score += ta036.sc;
  moved = moved || ta036.flg;

  const play_table &ta147 = pde[org.board[7]][org.board[4]][org.board[1]];
  dest->board[1] = ta147.p0;
  dest->board[4] = ta147.p1;
  dest->board[7] = ta147.p2;
  score += ta147.sc;
  moved = moved || ta147.flg;

  const play_table &ta258 = pde[org.board[8]][org.board[5]][org.board[2]];
  dest->board[2] = ta258.p0;
  dest->board[5] = ta258.p1;
  dest->board[8] = ta258.p2;
  score += ta258.sc;
  moved = moved || ta258.flg;
  dest->score = score;
  return moved;
}

inline bool play_rt(const state_t &org, state_t *dest) {
  bool moved = false;
  int score = org.score;
  const play_table &ta210 = pde[org.board[0]][org.board[1]][org.board[2]];
  dest->board[2] = ta210.p0;
  dest->board[1] = ta210.p1;
  dest->board[0] = ta210.p2;
  score += ta210.sc;
  moved = moved || ta210.flg;

  const play_table &ta543 = pde[org.board[3]][org.board[4]][org.board[5]];
  dest->board[5] = ta543.p0;
  dest->board[4] = ta543.p1;
  dest->board[3] = ta543.p2;
  score += ta543.sc;
  moved = moved || ta543.flg;

  const play_table &ta876 = pde[org.board[6]][org.board[7]][org.board[8]];
  dest->board[8] = ta876.p0;
  dest->board[7] = ta876.p1;
  dest->board[6] = ta876.p2;
  score += ta876.sc;
  moved = moved || ta876.flg;
  dest->score = score;
  return moved;
}

inline bool play_dw(const state_t &org, state_t *dest) {
  bool moved = false;
  int score = org.score;
  const play_table &ta630 = pde[org.board[0]][org.board[3]][org.board[6]];
  dest->board[6] = ta630.p0;
  dest->board[3] = ta630.p1;
  dest->board[0] = ta630.p2;
  score += ta630.sc;
  moved = moved || ta630.flg;

  const play_table &ta741 = pde[org.board[1]][org.board[4]][org.board[7]];
  dest->board[7] = ta741.p0;
  dest->board[4] = ta741.p1;
  dest->board[1] = ta741.p2;
  score += ta741.sc;
  moved = moved || ta741.flg;

  const play_table &ta852 = pde[org.board[2]][org.board[5]][org.board[8]];
  dest->board[8] = ta852.p0;
  dest->board[5] = ta852.p1;
  dest->board[2] = ta852.p2;
  score += ta852.sc;
  moved = moved || ta852.flg;
  dest->score = score;
  return moved;
}
inline bool play_lf(const state_t &org, state_t *dest) {
  bool moved = false;
  int score = org.score;
  const play_table &ta012 = pde[org.board[2]][org.board[1]][org.board[0]];
  dest->board[0] = ta012.p0;
  dest->board[1] = ta012.p1;
  dest->board[2] = ta012.p2;
  score += ta012.sc;
  moved = moved || ta012.flg;

  const play_table &ta345 = pde[org.board[5]][org.board[4]][org.board[3]];
  dest->board[3] = ta345.p0;
  dest->board[4] = ta345.p1;
  dest->board[5] = ta345.p2;
  score += ta345.sc;
  moved = moved || ta345.flg;

  const play_table &ta678 = pde[org.board[8]][org.board[7]][org.board[6]];
  dest->board[6] = ta678.p0;
  dest->board[7] = ta678.p1;
  dest->board[8] = ta678.p2;
  score += ta678.sc;
  moved = moved || ta678.flg;
  dest->score = score;
  return moved;
}

inline bool play_core(const state_t &org, state_t *dest, int i0, int i1, int i2,
                      int i3, int i4, int i5, int i6, int i7, int i8) {
  const play_table &ta012 = pde[org.board[i2]][org.board[i1]][org.board[i0]];
  const play_table &ta345 = pde[org.board[i5]][org.board[i4]][org.board[i3]];
  const play_table &ta678 = pde[org.board[i8]][org.board[i7]][org.board[i6]];
  if ((ta012.flg + ta345.flg + ta678.flg) == 0) return false;
  dest->board[i0] = ta012.p0;
  dest->board[i1] = ta012.p1;
  dest->board[i2] = ta012.p2;
  dest->board[i3] = ta345.p0;
  dest->board[i4] = ta345.p1;
  dest->board[i5] = ta345.p2;
  dest->board[i6] = ta678.p0;
  dest->board[i7] = ta678.p1;
  dest->board[i8] = ta678.p2;
  dest->score = org.score + ta012.sc + ta345.sc + ta678.sc;

  return true;
}

// 動く方向と現在の盤面と動いた後を格納するためのポインタ
inline bool play(int d, const state_t &org, state_t *dest) {
  switch (d) {
    case 0: /* UP */
      return play_core(org, dest, 0, 3, 6, 1, 4, 7, 2, 5, 8);
      break;
    case 1: /* RT */
      return play_core(org, dest, 2, 1, 0, 5, 4, 3, 8, 7, 6);
      break;
    case 2: /* DW */
      return play_core(org, dest, 6, 3, 0, 7, 4, 1, 8, 5, 2);
      break;
    case 3: /* LF */
      return play_core(org, dest, 0, 1, 2, 3, 4, 5, 6, 7, 8);
      break;
  }
  assert(false);
  return false;
}

inline bool gameOver(const state_t &org) {
  state_t ns;
  if (play(0, org, &ns)) return false;
  if (play(1, org, &ns)) return false;
  if (play(2, org, &ns)) return false;
  if (play(3, org, &ns)) return false;
  return true;
}

inline void print_state_and_ev(FILE *fp, int turn, const state_t &state,
                               const double ev[], int selected) {
  fprintf(fp, "state-ev {");
  fprintf(fp, "'turn':%d,'board':[%d", turn, state.board[0]);
  for (int i = 1; i < 9; i++) fprintf(fp, ",%d", state.board[i]);
  fprintf(fp, "],'score':%d,'evs':[%lf", state.score, ev[0]);
  for (int d = 1; d < 4; d++) fprintf(fp, ",%lf", ev[d]);
  fprintf(fp, "],'selected':%d}\n", selected);
}
