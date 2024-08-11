#include "act.h"
#include "knls.h"
#include "math.hpp"
#include "ten.h"

ten act(const ten &z, act_type act, dev_type dev)
{
    switch (act)
    {
    case RELU: {
        ten t_new = z;

        switch (dev)
        {
        case CPU: {
            for (auto i = 0; i < z.size; ++i)
                t_new.elem[i] = std::fmax(0.0f, z.elem[i]);

            return t_new;
        }
        default:
            std::cout << "Unknown dev." << std::endl;
            return ten();
        }
    }
    case SOFTMAX: {
        ten exp_scores = exp(z - max(z, 1), CPU);
        return exp_scores / sum(exp_scores, 1);
    }
    default:
        std::cout << "Unknown act." << std::endl;
        return ten();
    }
}