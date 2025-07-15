#include <algorithm>
#include <array>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <random>
#include <thread>

#include "Game2048_3_3.h"
using namespace std;
namespace fs = std::filesystem;

#ifdef NT4
#include "4tuples_sym.h"
string NT = "NT4";
#include "mcts_v2.hpp"
#else
#include "4tuples_sym.h"
string NT = "NT6";
#include "mcts_v2.hpp"
#endif

class GameOver {
 public:
  int gameover_turn;
  int game;
  int progress;
  int score;
  GameOver(int gameover_turn_init, int game_init, int progress_init,
           int score_init)
      : gameover_turn(gameover_turn_init),
        game(game_init),
        progress(progress_init),
        score(score_init) {}
};

int progress_calculation(int board[9]) {
  int sum = 0;
  for (int i = 0; i < 9; i++) {
    if (board[i] != 0) {
      sum += 1 << board[i];
    }
  }
  return sum / 2;
}

int main(int argc, char** argv) {
  if (argc < 10 + 1) {
    fprintf(stderr,
            "Usage: playgreedy <seed> <game_counts> <EV-file> "
            "<simulations> <randamTrun> <expand_count> <debug> <c> <Boltzmann> "
            "<expectimax>\n");
    exit(1);
  }

  int seed = atoi(argv[1]);
  int game_count = atoi(argv[2]);
  char* evfile = argv[3];
  int simulations = atoi(argv[4]);
  int randomTurn = atoi(argv[5]);
  int expand_count = atoi(argv[6]);
  bool debug = atoi(argv[7]);
  int c = atoi(argv[8]);
  bool Boltzmann = atoi(argv[9]);
  bool expectimax = atoi(argv[10]);

  string baseDir =
      "../board_data/MCTS" + std::string(1, evfile[0]) + "/" + "game_count" +
      std::to_string(game_count) + "_evfile" + std::string(evfile) +
      "_simulations" + std::to_string(simulations) + "_randomTurn" +
      std::to_string(randomTurn) + "_expandcount" +
      std::to_string(expand_count) + "_c" + std::to_string(c) + "_Boltzmann" +
      std::to_string(Boltzmann) + "_expectimax" + std::to_string(expectimax);

  // ディレクトリ作成
  try {
    if (!fs::exists(baseDir)) {
      fs::create_directories(baseDir);  // ネストされたディレクトリも作成
      cout << "Directory created: " << baseDir << endl;
    } else {
      cout << "Directory already exists: " << baseDir << endl;
    }
  } catch (const fs::filesystem_error& e) {
    cerr << "Error creating directory: " << e.what() << endl;
    return 1;
  }

  srand(seed);
  FILE* fp = fopen(evfile, "rb");
  if (fp == NULL) {
    fprintf(stderr, "cannot open file: %s\n", evfile);
    exit(1);
  }
  readEvs(fp);
  fclose(fp);

  mcts_searcher_t mcts_searcher(simulations, randomTurn, expand_count, c,
                                Boltzmann, expectimax, debug);

  list<array<int, 9>> state_list;
  list<array<int, 9>> after_state_list;
  list<GameOver> GameOver_list;
  list<array<double, 5>> eval_list;

  for (int gid = 0; gid < game_count; gid++) {
    state_t state = initGame();
    mcts_searcher.clearCache();

    for (int turn = 1;; turn++) {
      state_t copy;
      array<double, 5> evals = {-10000000000.0, -10000000000.0, -10000000000.0,
                                -10000000000.0, 0.0};
      int move = mcts_searcher.search(state, evals);

      state_t nextstate;
      bool result = play(move, state, &nextstate);
      assert(result);

      // 状態記録
      state_list.push_back(
          array<int, 9>{state.board[0], state.board[1], state.board[2],
                        state.board[3], state.board[4], state.board[5],
                        state.board[6], state.board[7], state.board[8]});

      after_state_list.push_back(array<int, 9>{
          nextstate.board[0], nextstate.board[1], nextstate.board[2],
          nextstate.board[3], nextstate.board[4], nextstate.board[5],
          nextstate.board[6], nextstate.board[7], nextstate.board[8]});

      // 評価値記録
      evals[4] = (double)progress_calculation(nextstate.board);
      eval_list.push_back(evals);

      putNewTile(&nextstate);

      if (gameOver(nextstate)) {
        GameOver_list.push_back(GameOver(turn, gid + 1,
                                         progress_calculation(nextstate.board),
                                         nextstate.score));
        break;
      }
      state = nextstate;
    }
  }

  // 出力処理
  ofstream stateFile(baseDir + "/state.txt");
  ofstream afterStateFile(baseDir + "/after-state.txt");
  ofstream evalFile(baseDir + "/eval.txt");

  auto trun_itr = GameOver_list.begin();
  int i = 0;
  for (auto itr = state_list.begin(); itr != state_list.end(); itr++) {
    i++;
    if (trun_itr != GameOver_list.end() && trun_itr->gameover_turn == i) {
      stateFile << "gameover_turn: " << trun_itr->gameover_turn
                << "; game: " << trun_itr->game
                << "; progress: " << trun_itr->progress
                << "; score: " << trun_itr->score << "\n";
      afterStateFile << "gameover_turn: " << trun_itr->gameover_turn
                     << "; game: " << trun_itr->game
                     << "; progress: " << trun_itr->progress
                     << "; score: " << trun_itr->score << "\n";
      evalFile << "gameover_turn: " << trun_itr->gameover_turn
               << "; game: " << trun_itr->game
               << "; progress: " << trun_itr->progress
               << "; score: " << trun_itr->score << "\n";
      trun_itr++;
      i = 0;
    } else {
      for (int j = 0; j < 9; j++) {
        stateFile << (*itr)[j] << " ";
      }
      stateFile << "\n";

      auto afterItr = after_state_list.begin();
      advance(afterItr, distance(state_list.begin(), itr));
      for (int j = 0; j < 9; j++) {
        afterStateFile << (*afterItr)[j] << " ";
      }
      afterStateFile << "\n";

      auto evalItr = eval_list.begin();
      advance(evalItr, distance(state_list.begin(), itr));
      for (int j = 0; j < 5; j++) {
        evalFile << std::fixed << (*evalItr)[j] << (j == 4 ? "" : " ");
      }
      evalFile << "\n";
    }
  }

  stateFile.close();
  afterStateFile.close();
  evalFile.close();

  return 0;
}
