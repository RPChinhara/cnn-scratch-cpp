#include "imdb.h"
#include "preproc.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

imdb load_imdb()
{
    std::ifstream file("datas/IMDB Dataset.csv");

    if (!file.is_open())
        std::cerr << "Failed to open the file." << std::endl;

    std::vector<std::string> reviews;
    std::vector<float> sentiments;

    std::string line;
    std::getline(file, line);

    size_t idx = 0;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;

        size_t startPos = 0;
        size_t endPosPositive = line.find(",positive");
        size_t endPosNegative = line.find(",negative");

        size_t endPos;
        if (endPosPositive != std::string::npos)
        {
            endPos = endPosPositive;
            sentiments.push_back(1.0f);
        }
        else if (endPosNegative != std::string::npos)
        {
            endPos = endPosNegative;
            sentiments.push_back(0.0f);
        }

        std::string text = line.substr(startPos, endPos - startPos);
        std::string text_no_link = regex_replace(text, R"((https?:\/\/|www\.)\S+)", "");
        std::string text_no_html = regex_replace(text_no_link, "<[^>]*>", " ");
        std::string text_sp_around_punc = regex_replace(text_no_html, "([.,!?-])", " $1 ");
        std::string text_no_consecutive_sp = regex_replace(text_sp_around_punc, "\\s{2,}", " ");
        std::string text_no_punc = regex_replace(text_no_consecutive_sp, "[\"#$%&'()*+/:;<=>@\\[\\\\\\]^_`{|}~]", " ");
        std::string text_no_num = regex_replace(text_no_punc, "\\d+", "");
        std::string text_no_ascii = regex_replace(text_no_num, "[^\\x00-\\x7f]", " ");
        std::string text_no_white_sp = regex_replace(text_no_ascii, "\\s+", " ");
        std::string text_no_emoji = regex_replace(text_no_white_sp, "[\xE2\x98\x80-\xE2\x9B\xBF]", "");
        std::string text_spell_corrected = regex_replace(text_no_emoji, "(.)\\1+", "$1$1");

        auto tokens = Tokenizer(text_spell_corrected);
        auto tokens_no_stop_words = RemoveStopWords(tokens);

        std::cout << "Text: " << idx + 1 << std::endl;
        std::cout << "++++++++++++++++++++++++++: " << std::endl;
        for (int i = 0; i < tokens_no_stop_words.size(); ++i)
        {
            std::cout << tokens_no_stop_words[i] << std::endl;
        }
        std::cout << "--------------------------: " << sentiments[idx] << std::endl << std::endl;
        ++idx;
    }

    return imdb();
}