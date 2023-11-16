#include "datasets.h"
#include "environment.h"
#include "nn.h"
#include "preprocessing.h"
#include "q_learning.h"
#include "window.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    // Create a console for logging otherwise I can't when WinMain() is used as the entry point because it doesn't use the standard console for input and output
    AllocConsole();
    freopen_s((FILE**)stdout, "CONOUT$", "w", stdout);

#if 0
    // Feedforward neural network
    // Load the Iris dataset
    Iris iris = load_iris();
    Tensor x = iris.features;
    Tensor y = iris.target;

    // Preprocess the dataset
    y = one_hot(y, 3);
    TrainTest2 train_temp = train_test_split(x, y, 0.2, 42);
    TrainTest2 val_test   = train_test_split(train_temp.x_second, train_temp.y_second, 0.5, 42);
    train_temp.x_first = min_max_scaler(train_temp.x_first);
    val_test.x_first   = min_max_scaler(val_test.x_first);
    val_test.x_second  = min_max_scaler(val_test.x_second);

    // Train and test the neural network
    NN nn = NN({ 4, 128, 3 }, 0.01f);
    nn.train(train_temp.x_first, train_temp.y_first, val_test.x_first, val_test.y_first);
    nn.predict(val_test.x_second, val_test.y_second);
#endif

#if 0
    // Reinforcement learning (Q-learining)
    Environment env = Environment();
    QLearning agent = QLearning(env.num_states, env.num_actions);

    unsigned int num_episodes = 1000;

    std::cout << "------------------- HEAD -------------------" << std::endl;

    for (int i = 0; i < num_episodes; ++i) {
        auto state = env.reset();
        bool done = false;
        int total_reward = 0;

        while (!done) {
            unsigned int action = agent.choose_action(state);
            std::cout << "action: " << action << std::endl;

            // Agent takes the selected action and observes the environment
            auto [next_state, reward, temp_done] = env.step(env.actions[action]);
            done = temp_done;

            // Agent updates the Q-table
            agent.update_q_table(state, action, reward, next_state);
            std::cout << agent.q_table << std::endl << std::endl;

            env.render();

            total_reward += reward;
            state = next_state;
        }
        std::cout << "Episode " << i + 1 << ": Total Reward = " << total_reward << std::endl << std::endl;
    }
#endif

    // Initialize the Windows application
    try {
        Window window(hInstance, nCmdShow);
        return window.messageLoop();
    } catch (const std::exception& e) {
        // Handle the error, e.g., show an error message
        MessageBox(nullptr, e.what(), "Error", MB_ICONERROR | MB_OK);
        return -1;
    }
}