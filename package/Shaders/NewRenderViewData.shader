Shader "Gaussian Splatting/NewRenderViewData"
{
    SubShader
    {
        Pass
        {
            // No culling or depth
            Cull Off
            ZWrite Off
            ZTest Always
            Blend SrcAlpha OneMinusSrcAlpha
            
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // #pragma require compute
            // #pragma use_dxc
            #pragma target 4.5

            #include "UnityCG.cginc"
            #include "NewGaussianSplatting.hlsl"

            float4 _VecScreenParams;
            int _NumRendered;
            StructuredBuffer<GeomData> _GeomData;
            StructuredBuffer<uint> _BinLeftPointListTileValue;
            StructuredBuffer<uint> _BinRightPointListTileValue;
            StructuredBuffer<uint2> _ImageLeftRange;
            StructuredBuffer<uint2> _ImageRightRange;
            // Texture2D _GSRenderTexture;
            
            struct v2f
            {
                float4 pos: SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            v2f vert (uint vtxID : SV_VertexID)
            {
                v2f o;
                float2 quadPos = float2(vtxID&1, (vtxID>>1)&1);
	            o.pos = float4(quadPos * 4.0 - 1.0, 1, 1);
                o.uv = float2(quadPos * 2.0) * _VecScreenParams;
                return o;
            }
            

            half4 frag (v2f i) : SV_Target
            {
                float W = _VecScreenParams.x;
                float H = _VecScreenParams.y;
                uint horizontal_blocks = (uint)((W + BLOCK_X - 1) / BLOCK_X);
                
                uint2 group_id = uint2(i.uv.x  / BLOCK_X, i.uv.y / BLOCK_Y);
                
                // 计算迭代次数
                uint2 range = _ImageLeftRange[group_id.y * horizontal_blocks + group_id.x];
                
                // 对非法值进行剔除，非法值包括：
                // 1. 错误的 _ImageRange 中的 x、y 值，超过了 _BinLeftPointListTileValue 的大小
                // 2. 错误的 _BinLeftPointListTileValue 中的 coll_id 值，超过了 _GeomData 
                // if (range.x >= _NumRendered || range.y >= _NumRendered)
                //     return;
                
                int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
                int toDo = range.y - range.x;
                
                if (rounds > 100)
                    return half4(1.0, 0.0, 0.0, 1.0);
                
                // 累计透明度、颜色和深度图输出
                float T = 1.0f;
                float3 color = 0.0f;
                
                // 遍历 Tile 中保存的高斯数据
                // 每个 thread 负责一个像素，逐像素累计高斯数据
                
                bool done = false;
                for (int j = 0; !done && j < toDo; j++)
                {
                    int coll_id = _BinLeftPointListTileValue[range.x + j];
                    // 根据像素到 2D 高斯中心点的距离，计算衰减度
                    float2 xy = _GeomData[coll_id].mean2D;
                    
                    float2 d = float2(xy.x - i.uv.x, xy.y - i.uv.y);
                
                    float4 con_o = _GeomData[coll_id].conic_opacity;
                    float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
                    if (power > 0.0f)
                        continue;
                
                    // 计算 alpha 透明度
                    float alpha = min(0.99f, con_o.w * exp(power));
                    if (alpha < 1.0f / 255.0f)
                        continue;
                
                    float test_T = T * (1 - alpha);
                    if (test_T < TEST_ALPHA)
                    {
                        done = true;
                        continue;
                    }
                
                    // 累加颜色
                    half3 tmp_color;
                    tmp_color.r = f16tof32(_GeomData[coll_id].rgb_depth.x >> 16);
                    tmp_color.g = f16tof32(_GeomData[coll_id].rgb_depth.x);
                    tmp_color.b = f16tof32(_GeomData[coll_id].rgb_depth.y >> 16);
                    color += tmp_color * alpha * T;
                
                    // // 累加深度
                    // inv_depth += (1 / collected_depth[j]) * alpha * T;
                    // // inv_depth += (1 / _GeomData[coll_id].depth) * alpha * T;
                    //
                    // // 更新累积 alpha
                    T = test_T;
                }

                // float4 color = _GSRenderTexture.Load(int3(i.uv, 0));
                
                return half4(GammaToLinearSpace(color), 1.0 - T);
            }
            ENDCG
        }
    }
}
