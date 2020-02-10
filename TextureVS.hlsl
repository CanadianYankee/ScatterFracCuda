static const float2 texCoords[4] = {
	{0.0f, 0.0f},
	{0.0f, 1.0f},
	{1.0f, 0.0f},
	{1.0f, 1.0f}
};

static const float4 posCoords[4] = {
	{-1.0f, -1.0f, 0.5f, 1.0f},
	{-1.0f, 1.0f, 0.5f, 1.0f},
	{1.0f, -1.0f, 0.5f, 1.0f},
	{1.0f, 1.0f, 0.5f, 1.0f}
};

struct VSOut
{
	float4 Pos : SV_POSITION;
	float2 Tex : TEXCOORD0;
};

VSOut TextureVS(uint vertexId : SV_VertexID)
{
	VSOut output;
	output.Tex = texCoords[vertexId];
	output.Pos = posCoords[vertexId];

	return output;
}
