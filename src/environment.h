#pragma once

#include <string>
#include <vector>

class Environment {
public:
    // States: hunger(the primary biological drive for eating is hunger, which is controlled by the hypothalamus in the brain. When the body's energy stores are depleted, signals are sent to the brain to initiate the feeling of hunger, motivating an individual to seek and consume food), thirst(dehydration or a decrease in the body's water content triggers the sensation of thirst), mental health, blood pressure, blood glucose level, fatigue, hygiene level, hair length, stress level, age, relationship, height and weight, energy level, sleepiness, health status(tooth decay, cancer, diabetes, emphysema, asthma), pain, social interactions, job satisfaction, clothing choices, emotional states, social media activity, weather conditions, temperature(日陰や雨よけ, tide), climate(wind) time of day, location, financial status (income, savings, and debt), education level, pains, feeling of goodness, news, nationality, homeostasis(such as blood sugar levels, electrolyte balance, and body temperature), emotions(such as stress, boredom, sadness, or happiness), self-esteem level(influences psychological well-being), auditory system, visual system, sense of smell, taste somatosensory system, goals(e.g., first and foremost simply survive, become leader of the group if he was a crocodile).

    // Actions: EAT(meat, vegetable), EXERCISE, SLEEP, SOCIALIZE (talk), hydrate, work(earn money), learn, get a haircut, brush teeth, take a bath/shower, grooming, play some sports, get sun, drinking, smoking, healthcare (go to the hospital, dentist...), shop, changing careers, moving to a new location, social media, entertainment (watch movies), transportation (how to commute or travel), clean the house, drive, move stuff, digestion, reproduction(such as mating, parenting, and ensuring the survival of their offspring).
    
   Environment() : actions({ "eat", "do_nothing", "up", "down", "left", "right" }), states({ "hungry", "neutral", "full" }) {
        num_states = states.size();
        num_actions = actions.size();
    }
    void render();
    int reset();
    std::tuple<int, int, bool> step(const std::string& action);
    std::vector<std::string> actions;
    int num_states;
    int num_actions;
private:
    int calculate_reward();
    bool check_termination();
    std::vector<std::string> states;
    int current_state = std::distance(states.begin(), std::find(states.begin(), states.end(), "neutral")); // index of neutral
    int days_lived = 0;
    int days_without_eating = 0;
    int max_days = 50;
    int max_days_without_eating = 43;
};