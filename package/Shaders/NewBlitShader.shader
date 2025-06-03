Shader "Hidden/New Gaussian Splatting/NewBlitShader"
{
    HLSLINCLUDE
        #pragma target 2.0
        #pragma editor_sync_compilation
        // Core.hlsl for XR dependencies
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
        #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"
         // Color.hlsl for color space conversion
        #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Color.hlsl"

        TEXTURE2D_X(_BlitDepth);
        uniform float4 _ScreenScale;
        
        float4 FragCustomBlit(Varyings input): SV_Target
        {
        #if defined(USE_TEXTURE2D_X_AS_ARRAY) && defined(BLIT_SINGLE_SLICE)
            return SAMPLE_TEXTURE2D_ARRAY_LOD(_BlitTexture, sampler_LinearClamp, input.texcoord.xy, _BlitTexArraySlice, _BlitMipLevel);
        #endif
        
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

            // const float2 uv = input.texcoord.xy;
            // const int kernel_size = 5;
            // const float sigma = 2.2;
            // float3 color_sum = 0;
            // float weight_sum = 0;
            //
            // for (int i = -1 * kernel_size; i < kernel_size; i++)
            // {
            //     for (int j = -1 * kernel_size; j < kernel_size; j++)
            //     {
            //         float2 varible = uv + float2(i * _BlitTexture_TexelSize.x, j * _BlitTexture_TexelSize.y);
            //         float factor = i * i + j * j;
            //         factor = (-factor) / (2 * sigma * sigma);
            //         float weight = 1/(sigma * sigma * 2 * PI) * exp(factor);
            //         //权重累积
            //         weight_sum += weight;
            //         color_sum += SAMPLE_TEXTURE2D_X_LOD(_BlitTexture, sampler_LinearClamp, varible, _BlitMipLevel).xyz * weight;
            //     }
            // }
            //
            // if(weight_sum > 0){
            //     //归一化，不然你会发现图片全部变白了
            //     color_sum = color_sum / weight_sum;
            // }

            float2 uv = input.texcoord.xy;
            float2 offset = (1.0 - _ScreenScale) * 0.5;
            uv = clamp(uv, offset + _BlitTexture_TexelSize, 1.0 - offset - _BlitTexture_TexelSize);
            return SAMPLE_TEXTURE2D_X_LOD(_BlitTexture, sampler_LinearClamp, uv, _BlitMipLevel);

            // return float4(SAMPLE_TEXTURE2D_X_LOD(_BlitDepth, sampler_LinearClamp, uv, 0)/10.0);
        }
    ENDHLSL
    
    SubShader
    {
        Tags{ "RenderPipeline" = "UniversalPipeline" }

        Pass
        {
            ZWrite Off ZTest Always Blend Off Cull Off
            Name "Bilinear"
            
            HLSLPROGRAM
                #pragma vertex Vert
                #pragma fragment FragCustomBlit
            ENDHLSL
        }
    }
}
