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

        float4 FragCustomBlit(Varyings input): SV_Target
        {
        #if defined(USE_TEXTURE2D_X_AS_ARRAY) && defined(BLIT_SINGLE_SLICE)
            return SAMPLE_TEXTURE2D_ARRAY_LOD(_BlitTexture, sampler_LinearClamp, input.texcoord.xy, _BlitTexArraySlice, _BlitMipLevel);
        #endif
        
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
            return SAMPLE_TEXTURE2D_X_LOD(_BlitTexture, sampler_LinearClamp, input.texcoord.xy, _BlitMipLevel);
            // return SAMPLE_TEXTURE2D_X_LOD(_BlitTexture, sampler_LinearClamp, uv, _BlitMipLevel);
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
