
#pragma once

#include <d3d11.h>

#include "renderer.h"

class mesh
{
public:
    bool initialize(renderer* r);

private:
    ID3D11Buffer* vertex_buffer;
};