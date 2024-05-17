#include <algorithm>
#include <iostream>
#include <set>
#include <string>

#include "lyrs.h"

int main()
{
    std::string text = "This is GeeksforGeeks a software training institute";

    std::sort(text.begin(), text.end());

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

    std::cout << unique_text << std::endl;

    std::vector<std::pair<char, int>> char_to_index;

    for (auto i = 0; i < unique_text.size(); ++i)
        char_to_index.emplace_back(unique_text[i], i);

    for (const auto &pair : char_to_index)
        std::cout << "'" << pair.first << "' -> " << pair.second << std::endl;

    return 0;
}