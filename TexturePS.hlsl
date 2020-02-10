struct VSOut
{
	float4 Pos : SV_POSITION;
	float2 Tex : TEXCOORD0;
};

texture2D g_Texture;
SamplerState g_Sampler;

float4 TexturePS(VSOut input) : SV_TARGET
{
	return g_Texture.Sample(g_Sampler, input.Tex);
}
