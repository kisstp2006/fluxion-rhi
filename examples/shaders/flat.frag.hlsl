struct Out { float4 position : SV_POSITION; float3 colour : COLOR0; };
float4 main(Out i) : SV_TARGET { return float4(i.colour, 1); }
