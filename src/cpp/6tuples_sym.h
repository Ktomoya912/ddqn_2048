#ifndef __6TUPLES_SYM_H__
#define __6TUPLES_SYM_H__
#pragma once
#include <climits>
#include <cmath>
#include <cstdlib>
using namespace std;

/**
 *  6タプルの定義．複数 cpp ファイルから #include は不可 (コード簡単化のため)
 */

#define TUPLE_SIZE 6
#define VARIATION_TILE 11
#define NUM_STAGES 2
#define ARRAY_LENGTH                                                   \
  (VARIATION_TILE * VARIATION_TILE * VARIATION_TILE * VARIATION_TILE * \
   VARIATION_TILE * VARIATION_TILE)

// #ifdef NT6
#define NUM_TUPLE 2
const int pos[NUM_TUPLE][TUPLE_SIZE] = {
    {0, 1, 2, 3, 4, 5},
    {0, 3, 4, 6, 7, 8},
};
// #else

double evs[NUM_STAGES][NUM_TUPLE][ARRAY_LENGTH] = {0};
double errs[NUM_STAGES][NUM_TUPLE][ARRAY_LENGTH] = {0};
double aerrs[NUM_STAGES][NUM_TUPLE][ARRAY_LENGTH] = {0};
int updatecounts[NUM_STAGES][NUM_TUPLE][ARRAY_LENGTH] = {0};

const int sympos[8][9] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8}, {0, 3, 6, 1, 4, 7, 2, 5, 8},
    {2, 1, 0, 5, 4, 3, 8, 7, 6}, {2, 5, 8, 1, 4, 7, 0, 3, 6},
    {6, 7, 8, 3, 4, 5, 0, 1, 2}, {6, 3, 0, 7, 4, 1, 8, 5, 2},
    {8, 7, 6, 5, 4, 3, 2, 1, 0}, {8, 5, 2, 7, 4, 1, 6, 3, 0},
};

inline void initEvs(double initEv) {
  double iev = initEv / NUM_TUPLE / 8;
  for (int s = 0; s < NUM_STAGES; s++) {
    for (int i = 0; i < NUM_TUPLE; i++) {
      for (int j = 0; j < ARRAY_LENGTH; j++) {
        evs[s][i][j] = iev;
      }
    }
  }
}

inline void writeEvs(FILE* fp) {
  size_t count;
  count =
      fwrite(evs, sizeof(double), NUM_STAGES * NUM_TUPLE * ARRAY_LENGTH, fp);
  if (count != NUM_STAGES * NUM_TUPLE * ARRAY_LENGTH) {
    fprintf(stderr, "in writeEvs(): written %ld elements (should be %d)\n",
            count, NUM_STAGES * NUM_TUPLE * ARRAY_LENGTH);
  }
  count =
      fwrite(errs, sizeof(double), NUM_STAGES * NUM_TUPLE * ARRAY_LENGTH, fp);
  if (count != NUM_STAGES * NUM_TUPLE * ARRAY_LENGTH) {
    fprintf(stderr, "in writeEvs(): written %ld elements (should be %d)\n",
            count, NUM_STAGES * NUM_TUPLE * ARRAY_LENGTH);
  }
  count =
      fwrite(aerrs, sizeof(double), NUM_STAGES * NUM_TUPLE * ARRAY_LENGTH, fp);
  if (count != NUM_STAGES * NUM_TUPLE * ARRAY_LENGTH) {
    fprintf(stderr, "in writeEvs(): written %ld elements (should be %d)\n",
            count, NUM_STAGES * NUM_TUPLE * ARRAY_LENGTH);
  }
  count = fwrite(updatecounts, sizeof(int),
                 NUM_STAGES * NUM_TUPLE * ARRAY_LENGTH, fp);
  if (count != NUM_STAGES * NUM_TUPLE * ARRAY_LENGTH) {
    fprintf(stderr, "in writeEvs(): written %ld elements (should be %d)\n",
            count, NUM_STAGES * NUM_TUPLE * ARRAY_LENGTH);
  }
}

inline void readEvs(FILE* fp) {
  size_t count;
  count = fread(evs, sizeof(double), NUM_STAGES * NUM_TUPLE * ARRAY_LENGTH, fp);
  if (count != NUM_STAGES * NUM_TUPLE * ARRAY_LENGTH) {
    fprintf(stderr, "in readEvs(): read %ld elements (should be %d)\n", count,
            NUM_STAGES * NUM_TUPLE * ARRAY_LENGTH);
  }
  count =
      fread(errs, sizeof(double), NUM_STAGES * NUM_TUPLE * ARRAY_LENGTH, fp);
  if (count != NUM_STAGES * NUM_TUPLE * ARRAY_LENGTH) {
    fprintf(stderr, "in readEvs(): read %ld elements (should be %d)\n", count,
            NUM_STAGES * NUM_TUPLE * ARRAY_LENGTH);
  }
  count =
      fread(aerrs, sizeof(double), NUM_STAGES * NUM_TUPLE * ARRAY_LENGTH, fp);
  if (count != NUM_STAGES * NUM_TUPLE * ARRAY_LENGTH) {
    fprintf(stderr, "in readEvs(): read %ld elements (should be %d)\n", count,
            NUM_STAGES * NUM_TUPLE * ARRAY_LENGTH);
  }
  count = fread(updatecounts, sizeof(int),
                NUM_STAGES * NUM_TUPLE * ARRAY_LENGTH, fp);
  if (count != NUM_STAGES * NUM_TUPLE * ARRAY_LENGTH) {
    fprintf(stderr, "in readEvs(): read %ld elements (should be %d)\n", count,
            NUM_STAGES * NUM_TUPLE * ARRAY_LENGTH);
  }
}

inline double calcEv(const int* board) {
  int s = 0;
  for (int i = 0; i < 9; i++) {
    if (board[i] >= 9) s = 1;
  }

  double ev = 0;
  for (int i = 0; i < NUM_TUPLE; i++) {
    for (int j = 0; j < 8; j++) {
      int index = 0;
      for (int k = 0; k < TUPLE_SIZE; k++) {
        index = index * VARIATION_TILE + board[sympos[j][pos[i][k]]];
      }
      ev += evs[s][i][index];
      // printf(" %f", evs[s][i][index]);
    }
  }
  // printf("\n");
  return ev;
}

inline double calcErr(const int* board) {
  int s = 0;
  for (int i = 0; i < 9; i++) {
    if (board[i] >= 9) s = 1;
  }

  double err = 0;
  for (int i = 0; i < NUM_TUPLE; i++) {
    for (int j = 0; j < 8; j++) {
      int index = 0;
      for (int k = 0; k < TUPLE_SIZE; k++) {
        index = index * VARIATION_TILE + board[sympos[j][pos[i][k]]];
      }
      err += errs[s][i][index];
    }
  }
  return err;
}

inline double calcAErr(const int* board) {
  int s = 0;
  for (int i = 0; i < 9; i++) {
    if (board[i] >= 9) s = 1;
  }

  double aerr = 0;
  for (int i = 0; i < NUM_TUPLE; i++) {
    for (int j = 0; j < 8; j++) {
      int index = 0;
      for (int k = 0; k < TUPLE_SIZE; k++) {
        index = index * VARIATION_TILE + board[sympos[j][pos[i][k]]];
      }
      aerr += aerrs[s][i][index];
    }
  }
  return aerr;
}

inline double calcMinAErr(const int* board) {
  int s = 0;
  for (int i = 0; i < 9; i++) {
    if (board[i] >= 9) s = 1;
  }

  double minAerr = DBL_MAX;
  for (int i = 0; i < NUM_TUPLE; i++) {
    for (int j = 0; j < 8; j++) {
      int index = 0;
      for (int k = 0; k < TUPLE_SIZE; k++) {
        index = index * VARIATION_TILE + board[sympos[j][pos[i][k]]];
      }
      if (minAerr > aerrs[s][i][index]) {
        minAerr = aerrs[s][i][index];
      }
    }
  }
  return minAerr;
}

inline int calcMinCount(const int* board) {
  int s = 0;
  for (int i = 0; i < 9; i++) {
    if (board[i] >= 9) s = 1;
  }

  int minCount = INT_MAX;
  for (int i = 0; i < NUM_TUPLE; i++) {
    for (int j = 0; j < 8; j++) {
      int index = 0;
      for (int k = 0; k < TUPLE_SIZE; k++) {
        index = index * VARIATION_TILE + board[sympos[j][pos[i][k]]];
      }
      if (minCount > updatecounts[s][i][index]) {
        minCount = updatecounts[s][i][index];
      }
    }
  }
  return minCount;
}

inline void update(const int* board, double diff) {
  diff = diff / NUM_TUPLE / 8;
  int s = 0;
  for (int i = 0; i < 9; i++) {
    if (board[i] >= 9) s = 1;
  }

  for (int i = 0; i < NUM_TUPLE; i++) {
    for (int j = 0; j < 8; j++) {
      int index = 0;
      for (int k = 0; k < TUPLE_SIZE; k++) {
        index = index * VARIATION_TILE + board[sympos[j][pos[i][k]]];
      }
      aerrs[s][i][index] += fabs(diff);
      errs[s][i][index] += diff;
      if (aerrs[s][i][index] == 0) {
        evs[s][i][index] += diff;
      } else {
        evs[s][i][index] +=
            diff * (fabs(errs[s][i][index]) / aerrs[s][i][index]);
      }
      updatecounts[s][i][index]++;
    }
  }
}

#endif  // __TUPLES6_SYM_H__
