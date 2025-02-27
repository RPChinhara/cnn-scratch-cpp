struct PS_INPUT {
    float4 pos : SV_POSITION;
};

float4 main(PS_INPUT input) : SV_TARGET {
    return float4(0.0f, 0.5f, 1.0f, 1.0f); // Simple solid blue color
}

// fxc /T ps_5_0 /E main /Fo shaders/compiled/basic_ps.cso shaders/src/basic_ps.hlsl