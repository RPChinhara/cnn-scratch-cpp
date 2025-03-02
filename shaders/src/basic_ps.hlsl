struct PS_INPUT {
    float4 pos : SV_POSITION;
    float4 color : COLOR;
};

float4 main(PS_INPUT input) : SV_TARGET {
    return input.color;  // Use the color passed from VS
}

// fxc /T ps_5_0 /E main /Fo shaders/compiled/basic_ps.cso shaders/src/basic_ps.hlsl