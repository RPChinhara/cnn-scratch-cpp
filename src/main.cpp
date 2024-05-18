#include <algorithm>
#include <iostream>
#include <set>
#include <string>

#include "lyrs.h"

std::vector<std::pair<char, size_t>> map_char_index(const std::string &text)
{
    std::vector<std::pair<char, size_t>> char_to_index;

    for (auto i = 0; i < text.size(); ++i)
        char_to_index.emplace_back(text[i], i);

    return char_to_index;
}

std::string rm_duplicates(const std::string &text)
{
    std::set<char> seen;
    std::string unique_text;

    for (auto ch : text)
    {
        if (seen.find(ch) == seen.end())
        {
            seen.insert(ch);
            unique_text += ch;
        }
    }

    return unique_text;
}

int main()
{
    std::string text = "This is GeeksforGeeks a software training institute";

    std::sort(text.begin(), text.end());

    auto unique_text = rm_duplicates(text);
    auto char_to_index = map_char_index(unique_text);

    for (const auto &pair : char_to_index)
        std::cout << "'" << pair.first << "' -> " << pair.second << std::endl;

    return 0;
}