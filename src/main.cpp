#include <algorithm> // for std::sort
#include <iostream>
#include <set>
#include <string> // for std::string

#include "lyrs.h"

int main()
{
    std::string text = "This is GeeksforGeeks a software training institute";

    // Use std::sort to sort the string in alphabetical order
    std::sort(text.begin(), text.end());

    std::set<char> seen;
    std::string unique_text;

    for (auto ch : text)
    {
        if (seen.find(ch) == seen.end())
        {
            seen.insert(ch);
            unique_text += ch; // Append the original character
        }
    }

    std::cout << unique_text << std::endl;

    // Vector to store character-integer pairs
    std::vector<std::pair<char, int>> char_to_index;

    // Assign integer to each character
    for (auto i = 0; i < unique_text.size(); ++i)
    {
        char_to_index.emplace_back(unique_text[i], i);
    }

    // Output the character-integer pairs
    for (const auto &pair : char_to_index)
    {
        std::cout << "'" << pair.first << "' -> " << pair.second << std::endl;
    }
    
    return 0;
}