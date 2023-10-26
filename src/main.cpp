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
    // states - hunger(start with negative hunger?), thirstiness, mental health, blood pressure, blood glucose level, hygiene level, hair length, stress level, age, relationship, height and weight, energy level, sleepiness, health status(tooth decay, cancer, diabetes, emphysema, asthma), pain, social interactions, job satisfaction, clothing choices, emotional states, social media activity, weather conditions, temperature, time of day, location, financial status (income, savings, and debt), education level.
    // first start with 3 states which are hunger, thirstiness, and mental health each have 5 states (very hungry, hungry, neutral, full, very full), (very thirsty, thirsty, neutral, satisfied, very satisfied), (very stressed, stressed, neutral, content, happy) which means 5 (hunger) * 5 (thirstiness) * 5 (mental health) = 125 states.
    unsigned int num_states = 125;

    // actions - EAT(meat, vegetable), EXERCISE, SLEEP, SOCIALIZE, hydrate, work(earn money), learn, get a haircut, brush teeth, take a bath/shower, grooming, play some sports, get sun, drinking, smoking, healthcare (go to the hospital, dentist...), shop, changing careers, moving to a new location, social media, entertainment (watch movies), transportation (how to commute or travel)
    // first start with 4 actions which are eat, exercise, sleep, socialize. Can perform these actions at various levels of intensity which are 3 levels (low, medium, high). In this case, it would be 3 levels for each of the 4 actions, resulting in a total of 3^4 = 81 possible action combinations.
    // TODO: I could go more detail e.g., 
    // Food Type: Specify the type of food the agent can choose to eat, such as healthy options (vegetables, fruits, lean proteins) or unhealthy options (fast food, sugary snacks).
    // Portion Size: Define different portion sizes (small, medium, large) for the agent's meals.
    // Meal Timing: Determine when the agent can eat (breakfast, lunch, dinner, snacks).
    unsigned int num_actions = 81;

    // Q-table would have 125 (states) * 81 (action combinations) rows and columns, resulting in a total of 10,125 cells in the table.
    // | State (Hunger, Thirstiness, Mental Health) | Eat (Low) | Eat (Medium) | Eat (High) | Exercise (Low) | ... |
    // |--------------------------------------------|-----------|--------------|------------|----------------|-----|
    // | (Very Hungry, Very Thirsty, Stressed)      | ?         | ?            | ?          | ?              | ... |
    // | (Very Hungry, Very Thirsty, Neutral)       | ?         | ?            | ?          | ?              | ... |
    // | ...                                        | ...       | ...          | ...        | ...            | ... |
    // | (Full, Very Satisfied, Happy)              | ?         | ?            | ?          | ?              | ... |
    // | (Very Full, Very Satisfied, Happy)         | ?         | ?            | ?          | ?              | ... |
    
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
    // Environment env = Environment("hello");

    // std::string state = env.reset();
    // env.render();

    // // Example action: guessing the letter "h" for the 0th position
    // auto result = env.step({0, 'h'});
    // std::cout << "next_state: " << std::get<0>(result) << " reward: " << std::get<1>(result) << " done: " << std::get<2>(result) << std::endl;
    // env.render();

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