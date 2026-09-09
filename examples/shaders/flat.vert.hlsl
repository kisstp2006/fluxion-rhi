struct In { float2 position : ATTR0; float3 colour : ATTR1; };
struct Out { float4 position : SV_POSITION; float3 colour : COLOR0; };
Out main(In i) { Out o; o.position = float4(i.position, 0, 1); o.colour = i.colour; return o; }
