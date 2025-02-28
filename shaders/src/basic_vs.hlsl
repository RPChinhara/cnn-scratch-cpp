struct VS_INPUT {
    float3 pos : POSITION;
};

struct VS_OUTPUT {
    float4 pos : SV_POSITION;
};

cbuffer TransformBuffer : register(b0) {
    float4x4 wvp;
};

VS_OUTPUT main(VS_INPUT input) {
    VS_OUTPUT output;
    // output.pos = mul(float4(input.pos, 1.0f), wvp);
    output.pos = float4(input.pos, 1.0f);  // For a simple rectagle - Pass position to clip space (identity pass-through)
    return output;
}

// fxc /T vs_5_0 /E main /Fo shaders/compiled/basic_vs.cso shaders/src/basic_vs.hlsl