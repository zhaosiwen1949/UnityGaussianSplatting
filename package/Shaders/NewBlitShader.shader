Shader "Hidden/New Gaussian Splatting/NewBlitShader"
{
    HLSLINCLUDE
        #pragma target 4.5
        #pragma editor_sync_compilation
        // 
        #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
        #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Filtering.hlsl"
        // Core.hlsl for XR dependencies
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
        #include "Packages/com.unity.render-pipelines.universal/Shaders/PostProcessing/Common.hlsl"
        #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"
         // Color.hlsl for color space conversion
        #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Color.hlsl"

        TEXTURE2D_X(_BlitDepth);
        float4 _ScreenScale;
        float4 _SourceSize;
        int _ShowDepth;

        #define FSR_INPUT_TEXTURE _BlitTexture
        #define FSR_INPUT_SAMPLER sampler_LinearClamp

        #include "Packages/com.unity.render-pipelines.core/Runtime/PostProcessing/Shaders/FSRCommon.hlsl"

        half4 FragEASU(Varyings input) : SV_Target
        {
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

            float2 uv = UnityStereoTransformScreenSpaceTex(input.texcoord);
            uint2 integerUv = uv * _ScreenParams.xy;

            half3 color = ApplyEASU(integerUv);

            return half4(color, 1.0);
        }

        half4 FragRCAS(Varyings input) : SV_Target
        {
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

            float2 uv = UnityStereoTransformScreenSpaceTex(input.texcoord);
            float4 offset = (1.0 - _ScreenScale) * 0.5;
            float2 down_limit = offset.xw + _BlitTexture_TexelSize.xy;
            float2 up_limit = 1.0 - offset.yz - _BlitTexture_TexelSize.xy;

            half4 color = 0.0;
            #if defined(SHOW_DEPTH)
            uv = clamp(uv, down_limit, up_limit);
            color = half4(SAMPLE_TEXTURE2D_X_LOD(_BlitDepth, sampler_LinearClamp, uv, 0)/10.0);
            #else
            if (uv.x <= down_limit.x || uv.y <= down_limit.y || uv.x >= up_limit.x || uv.y >= up_limit.y)
            {
                uv = clamp(uv, down_limit, up_limit);
                color = SAMPLE_TEXTURE2D_X_LOD(_BlitTexture, sampler_LinearClamp, uv, _BlitMipLevel);
            }
            else
            {
                int2 positionSS  = uv * _SourceSize.xy;
                color = half4(ApplyRCAS(positionSS), 1.0);
            }
            
            // // Bilinear
            // uv = clamp(uv, down_limit, up_limit);
            // color = half4(SAMPLE_TEXTURE2D_X_LOD(_BlitTexture, sampler_LinearClamp, uv, 0));
            #endif

            return color;
        }
        
        float4 FragCustomBlit(Varyings input): SV_Target
        {
        #if defined(USE_TEXTURE2D_X_AS_ARRAY) && defined(BLIT_SINGLE_SLICE)
            return SAMPLE_TEXTURE2D_ARRAY_LOD(_BlitTexture, sampler_LinearClamp, input.texcoord.xy, _BlitTexArraySlice, _BlitMipLevel);
        #endif
        
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

            float2 uv = input.texcoord.xy;
            float2 offset = (1.0 - _ScreenScale) * 0.5;
            uv = clamp(uv, offset + _BlitTexture_TexelSize, 1.0 - offset - _BlitTexture_TexelSize);
            return SAMPLE_TEXTURE2D_X_LOD(_BlitTexture, sampler_LinearClamp, uv, _BlitMipLevel);

            // return float4(SAMPLE_TEXTURE2D_X_LOD(_BlitDepth, sampler_LinearClamp, uv, 0)/10.0);
        }
        
    ENDHLSL
    
    SubShader
    {
        Tags{ "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" }
        LOD 100
        ZWrite Off ZTest Always Blend Off Cull Off

        Pass
        {
            Name "Bilinear"
            
            HLSLPROGRAM
                #pragma vertex Vert
                #pragma fragment FragCustomBlit
            ENDHLSL
        }

        Pass
        {
            Name "EASU"

            HLSLPROGRAM
                #pragma vertex Vert
                #pragma fragment FragEASU
            ENDHLSL
        }

        Pass
        {
            Name "RCAS"

            HLSLPROGRAM
                #pragma vertex Vert
                #pragma fragment FragRCAS
                #pragma multi_compile_local __ FSR_RCAS_DENOISE SHOW_DEPTH
            ENDHLSL
        }
    }
}
