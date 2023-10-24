#include "datasets.h"
#include "nn.h"
#include "preprocessing.h"
#include "q_learning.h"
#include "environment.h"
#include "window.h"

#include <random>

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    // Create a console for logging otherwise I can't when WinMain() is used as the entry point because it doesn't use the standard console for input and output
    AllocConsole();
    freopen_s((FILE**)stdout, "CONOUT$", "w", stdout);

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

    // Train and test neural network
    NN nn = NN({ 4, 128, 3 }, 0.01f);
    nn.train(train_temp.x_first, train_temp.y_first, val_test.x_first, val_test.y_first);
    nn.predict(val_test.x_second, val_test.y_second);

    // Q-learining
    // states
    // - hunger(start with negative hunger?), thirstiness, mental health, blood pressure, blood glucose level, odor, hair length, stress level, age, relationship, height and weight, energy level, sleepiness, health status(cacer, diabetes, emphysema, asthma), pain
    // - weather conditions, time of day
    // - saving, debt
    // first start with three states which are hunger, thirstiness, and mental health each have 3 states (low, medium, high) which means result in 3 * 3 * 3 = 9 states.
    unsigned int num_states   = 9;

    // actions - EAT(meat, vegetable), EXERCISE, SLEEP, hydrate, work(earn money), study, get a haircut, brush teeth, take a bath/shower, play some sports, get sun, drinking, smoking, check health status
    unsigned int num_actions  = 1000;

    unsigned int num_episodes = 1000;
    QLearning agent = QLearning(num_states, num_actions);

    for (int i = 0; i < num_episodes; ++i) {
        std::random_device rd;
        std::mt19937 gen(rd());
        auto state = std::uniform_int_distribution<unsigned int>(0, 5 - 1); // Start with a random state
        bool done  = false;

        while (!done) {
            unsigned int action = agent.choose_action(state(gen));

            // Here you would take the action in the environment and get the next_state and reward.
            // This is just a placeholder example:
            auto next_state = std::uniform_int_distribution<unsigned int>(0, 5 - 1);
            auto reward = -1 ? action != 2 : 1; // Assume action 2 is the "correct" action for demonstration

            agent.update(state(gen), action, reward, next_state(gen));

            state = next_state;

            // Just an example to end the loop
            if (reward == 1) {
                done = true;
            }
        }
    }

    std::cout << agent.q_table << std::endl;

    // Using the environment:
    Environment env = Environment("hello");

    std::string state = env.reset();
    env.render();

    // Example action: guessing the letter "h" for the 0th position
    auto result = env.step({0, 'h'});
    std::cout << "next_state: " << std::get<0>(result) << " reward: " << std::get<1>(result) << " done: " << std::get<2>(result) << std::endl;
    env.render();

    // Making the window
    try {
        Window window(hInstance, nCmdShow);
        return window.messageLoop();
    } catch (const std::exception& e) {
        // Handle the error, e.g., show an error message
        MessageBox(NULL, e.what(), "Error", MB_ICONERROR | MB_OK);
        return -1;
    }
}