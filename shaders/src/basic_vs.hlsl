cbuffer TransformBuffer : register(b0) {
    float4x4 wvp;
    float4 objectColor;
};

struct VS_INPUT {
    float3 pos : POSITION;
};

struct VS_OUTPUT {
    float4 pos : SV_POSITION;
    float4 color : COLOR;  // Add this line
};

VS_OUTPUT main(VS_INPUT input) {
    VS_OUTPUT output;
    output.pos = mul(float4(input.pos, 1.0f), wvp);
    output.color = objectColor;
    return output;
}

// fxc /T vs_5_0 /E main /Fo shaders/compiled/basic_vs.cso shaders/src/basic_vs.hlsl