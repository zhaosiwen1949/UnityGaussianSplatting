// SPDX-License-Identifier: MIT
#if GS_ENABLE_URP

#if !UNITY_6000_0_OR_NEWER
#error Unity Gaussian Splatting URP support only works in Unity 6 or later
#endif

using Unity.Mathematics;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.XR;

namespace GaussianSplatting.Runtime
{
    // Note: I have no idea what is the purpose of ScriptableRendererFeature vs ScriptableRenderPass, which one of those
    // is supposed to do resource management vs logic, etc. etc. Code below "seems to work" but I'm just fumbling along,
    // without understanding any of it.
    //
    // ReSharper disable once InconsistentNaming
    class NewGaussianSplatURPFeature : ScriptableRendererFeature
    {
        class NewGSRenderPass : ScriptableRenderPass
        {
            const string GaussianSplatRTName = "_GaussianSplatRT";
            const string BlitRTName = "_BlitRT";
            const string CurrentGaussianSplatDepthName = "_CurrentGaussianSplatDepth";
            const string PreGaussianSplatDepthName = "_PreGaussianSplatDepth";

            const string ProfilerTag = "GaussianSplatRenderGraph";
            static readonly ProfilingSampler s_profilingSampler = new(ProfilerTag);
            static readonly int s_gaussianSplatRT = Shader.PropertyToID(GaussianSplatRTName);
            
            public Material blitMaterial { get; set; }

            private float TextureScale = 1.0f;
            private RTHandle m_preDepthTexture;
            private RTHandle m_currentDepthTexture;
            private XRSettings.StereoRenderingMode depthStereoRenderingMode = XRSettings.StereoRenderingMode.MultiPass;
            private Matrix4x4 m_preViewProjectionMatrix = Matrix4x4.identity;
            private Matrix4x4 m_preLeftViewProjectionMatrix = Matrix4x4.identity;
            private Matrix4x4 m_preRightViewProjectionMatrix = Matrix4x4.identity;
            
            class PassData
            {
                internal UniversalCameraData CameraData;
                internal Material BlitMaterial;
                internal float2 ScreenScale;
                internal float Sharpness;
                internal TextureHandle SourceTexture;
                internal TextureHandle SourceDepth;
                internal TextureHandle GaussianSplatRT;
                internal TextureHandle BlitRT;
                internal TextureHandle PreGaussianSplatDepth;
                internal TextureHandle CurrentGaussianSplatDepth;
                internal Matrix4x4 PreLeftViewProjectionMatrix;
                internal Matrix4x4 PreRightViewProjectionMatrix;
            }
            

            public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
            {
                using var builder = renderGraph.AddUnsafePass(ProfilerTag, out PassData passData);

                var cameraData = frameData.Get<UniversalCameraData>();
                var resourceData = frameData.Get<UniversalResourceData>();
                TextureScale = NewGaussianSplatRenderSystem.instance.GetTextureScale();

                RenderTextureDescriptor rtDesc = cameraData.cameraTargetDescriptor;
                rtDesc.width = (int)(rtDesc.width * TextureScale);
                rtDesc.height = (int)(rtDesc.height * TextureScale);
                rtDesc.depthBufferBits = 0;
                rtDesc.msaaSamples = 1;
                rtDesc.autoGenerateMips = false;
                rtDesc.graphicsFormat = GraphicsFormat.R16G16B16A16_SFloat;
                rtDesc.enableRandomWrite = true;
                var textureHandle = UniversalRenderer.CreateRenderGraphTexture(renderGraph, rtDesc, GaussianSplatRTName, true);
                
                RenderTextureDescriptor blitRTDesc = cameraData.cameraTargetDescriptor;
                blitRTDesc.depthBufferBits = 0;
                blitRTDesc.msaaSamples = 1;
                blitRTDesc.autoGenerateMips = false;
                blitRTDesc.graphicsFormat = GraphicsFormat.R16G16B16A16_SFloat;
                var blitTextureHandle = UniversalRenderer.CreateRenderGraphTexture(renderGraph, blitRTDesc, BlitRTName, true);
                
                // 初始化创建 PreDepthTexture 和 CurrentDepthTexture
                if (m_preDepthTexture == null || m_currentDepthTexture == null || depthStereoRenderingMode != XRSettings.stereoRenderingMode)
                {
                    depthStereoRenderingMode = XRSettings.stereoRenderingMode;
                    
                    // RenderTextureDescriptor depthDesc = new RenderTextureDescriptor((int)(rtDesc.width * TextureScale), (int)(rtDesc.height * TextureScale));
                    RenderTextureDescriptor depthDesc = cameraData.cameraTargetDescriptor;
                    depthDesc.width = (int)(depthDesc.width * TextureScale);
                    depthDesc.height = (int)(depthDesc.height * TextureScale);
                    depthDesc.depthBufferBits = 0;
                    depthDesc.msaaSamples = 1;
                    depthDesc.autoGenerateMips = false;
                    depthDesc.graphicsFormat = GraphicsFormat.R16_SFloat;
                    depthDesc.enableRandomWrite = true;
                    depthDesc.vrUsage = depthStereoRenderingMode == XRSettings.StereoRenderingMode.SinglePassInstanced ? VRTextureUsage.TwoEyes : VRTextureUsage.None;
                    RenderingUtils.ReAllocateIfNeeded(ref m_preDepthTexture, depthDesc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: PreGaussianSplatDepthName );
                    RenderingUtils.ReAllocateIfNeeded(ref m_currentDepthTexture, depthDesc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: CurrentGaussianSplatDepthName );
                }
                
                // 交换 PreDepthTexture 和 CurrentDepthTexture
                (m_preDepthTexture, m_currentDepthTexture) = (m_currentDepthTexture, m_preDepthTexture);

                // 引入 PreDepthTexture 和 CurrentDepthTexture
                TextureHandle preDepthTextureHandle = renderGraph.ImportTexture(m_preDepthTexture);
                TextureHandle currentDepthTextureHandle = renderGraph.ImportTexture(m_currentDepthTexture);
                
                passData.CameraData = cameraData;
                passData.BlitMaterial = blitMaterial;
                passData.ScreenScale = new float2(NewGaussianSplatRenderSystem.instance.GetWidthScale(),
                    NewGaussianSplatRenderSystem.instance.GetHeightScale());
                passData.Sharpness = NewGaussianSplatRenderSystem.instance.GetSharpness();
                passData.SourceTexture = resourceData.activeColorTexture;
                passData.SourceDepth = resourceData.activeDepthTexture;
                passData.GaussianSplatRT = textureHandle;
                passData.BlitRT = blitTextureHandle;
                passData.PreGaussianSplatDepth = preDepthTextureHandle;
                passData.CurrentGaussianSplatDepth = currentDepthTextureHandle;
                if (XRSettings.stereoRenderingMode == XRSettings.StereoRenderingMode.SinglePassInstanced)
                {
                    passData.PreLeftViewProjectionMatrix = m_preLeftViewProjectionMatrix;
                    passData.PreRightViewProjectionMatrix = m_preRightViewProjectionMatrix;
                }
                else
                {
                    passData.PreLeftViewProjectionMatrix = m_preViewProjectionMatrix;
                    passData.PreRightViewProjectionMatrix = Matrix4x4.identity;
                }
                
                // 更新 m_preViewProjectionMatrix
                Camera camera = cameraData.camera;
                if (XRSettings.stereoRenderingMode == XRSettings.StereoRenderingMode.SinglePassInstanced)
                {
                    Matrix4x4 matLView = camera.GetStereoViewMatrix(Camera.StereoscopicEye.Left);
                    Matrix4x4 matRView = camera.GetStereoViewMatrix(Camera.StereoscopicEye.Right);
                    camera.CopyStereoDeviceProjectionMatrixToNonJittered(Camera.StereoscopicEye.Left);
                    Matrix4x4 matLProj =
                        GL.GetGPUProjectionMatrix(camera.GetStereoNonJitteredProjectionMatrix(Camera.StereoscopicEye.Left), true);
                    camera.CopyStereoDeviceProjectionMatrixToNonJittered(Camera.StereoscopicEye.Right);
                    Matrix4x4 matRProj =
                        GL.GetGPUProjectionMatrix(camera.GetStereoNonJitteredProjectionMatrix(Camera.StereoscopicEye.Right), true);
                    m_preLeftViewProjectionMatrix = matLProj * matLView;
                    m_preRightViewProjectionMatrix = matRProj * matRView;
                }
                else
                {
                    Matrix4x4 currentViewMatrix = camera.worldToCameraMatrix;
                    Matrix4x4 currentGLProjectionMatrix = GL.GetGPUProjectionMatrix(camera.projectionMatrix, true);
                    m_preViewProjectionMatrix = currentGLProjectionMatrix * currentViewMatrix;
                }

                builder.UseTexture(resourceData.activeColorTexture, AccessFlags.ReadWrite);
                builder.UseTexture(resourceData.activeDepthTexture);
                builder.UseTexture(textureHandle, AccessFlags.Write);
                builder.UseTexture(blitTextureHandle, AccessFlags.Write);
                builder.UseTexture(preDepthTextureHandle);
                builder.UseTexture(currentDepthTextureHandle, AccessFlags.Write);
                
                builder.AllowPassCulling(false);
                builder.SetRenderFunc(static (PassData data, UnsafeGraphContext context) =>
                {
                    var commandBuffer = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
                    using var _ = new ProfilingScope(commandBuffer, s_profilingSampler);
                    
                    // Tile 渲染方法
                    NewGaussianSplatRenderSystem.instance.TileStereoRenderSplats(
                        data.CameraData.camera,
                        commandBuffer,
                        data.GaussianSplatRT,
                        data.PreGaussianSplatDepth,
                        data.CurrentGaussianSplatDepth,
                        data.PreLeftViewProjectionMatrix,
                        data.PreRightViewProjectionMatrix
                    );
                    
                    // NewGaussianSplatRenderSystem.instance.TileSingleRenderSplats(
                    //     data.CameraData.camera,
                    //     commandBuffer,
                    //     data.GaussianSplatRT,
                    //     data.PreGaussianSplatDepth,
                    //     data.CurrentGaussianSplatDepth,
                    //     data.PreLeftViewProjectionMatrix,
                    //     data.PreRightViewProjectionMatrix
                    // );
                    
                    commandBuffer.BeginSample(NewGaussianSplatRenderSystem.s_ProfCompose);
                    commandBuffer.SetFoveatedRenderingMode(FoveatedRenderingMode.Enabled);
                    BlitCameraTexture(commandBuffer, 
                        data.GaussianSplatRT, 
                        data.SourceTexture, 
                        data.CurrentGaussianSplatDepth,
                        data.BlitRT,
                        data.BlitMaterial, 
                        data.ScreenScale,
                        data.Sharpness);
                    commandBuffer.EndSample(NewGaussianSplatRenderSystem.s_ProfCompose);
                });
            }

            public void CleanUp()
            {
                m_preDepthTexture?.Release();
                m_currentDepthTexture?.Release();
            }
            
            static MaterialPropertyBlock s_PropertyBlock = new MaterialPropertyBlock();
            public static readonly int BlitTexture = Shader.PropertyToID("_BlitTexture");
            public static readonly int BlitScaleBias = Shader.PropertyToID("_BlitScaleBias");
            public static readonly int BlitDepth = Shader.PropertyToID("_BlitDepth");
            public static readonly int ScreenScale = Shader.PropertyToID("_ScreenScale");
            public static readonly int _SourceSize = Shader.PropertyToID("_SourceSize");
            private static void BlitCameraTexture(
                CommandBuffer cmd,
                RTHandle source,
                RTHandle destination,
                RTHandle depth,
                RTHandle blitRT,
                Material material,
                float2 screen_scale,
                float sharpness)
            {
                Vector2 viewportScale = source.useScaling ? new Vector2(source.rtHandleProperties.rtHandleScale.x, source.rtHandleProperties.rtHandleScale.y) : Vector2.one;
                // Will set the correct camera viewport as well.
                CoreUtils.SetRenderTarget(cmd, blitRT);
                s_PropertyBlock.SetVector(BlitScaleBias, viewportScale);
                s_PropertyBlock.SetTexture(BlitTexture, source);
                s_PropertyBlock.SetTexture(BlitDepth, depth);
                s_PropertyBlock.SetVector(ScreenScale, new Vector2(screen_scale.x, screen_scale.y));
                
                // EASU
                var fsrInputSize = new Vector2(source.referenceSize.x, source.referenceSize.y);
                var fsrOutputSize = new Vector2(blitRT.referenceSize.x, blitRT.referenceSize.y);
                FSRUtils.SetEasuConstants(cmd, fsrInputSize, fsrInputSize, fsrOutputSize);
                
                cmd.DrawProcedural(Matrix4x4.identity, material, 1, MeshTopology.Triangles, 3, 1, s_PropertyBlock);
                
                // RCAS
                Vector2 blitViewportScale = blitRT.useScaling ? new Vector2(blitRT.rtHandleProperties.rtHandleScale.x, blitRT.rtHandleProperties.rtHandleScale.y) : Vector2.one;
                CoreUtils.SetRenderTarget(cmd, destination);
                s_PropertyBlock.SetVector(BlitScaleBias, blitViewportScale);
                s_PropertyBlock.SetTexture(BlitTexture, blitRT);
                float width = blitRT.rt.width;
                float height = blitRT.rt.height;
                if (blitRT.rt.useDynamicScale)
                {
                    width *= ScalableBufferManager.widthScaleFactor;
                    height *= ScalableBufferManager.heightScaleFactor;
                }
                cmd.SetGlobalVector(_SourceSize, new Vector4(width, height, 1.0f / width, 1.0f / height));
                FSRUtils.SetRcasConstantsLinear(cmd, sharpness);
                material.EnableKeyword("FSR_RCAS_DENOISE");
                
                cmd.DrawProcedural(Matrix4x4.identity, material, 2, MeshTopology.Triangles, 3, 1, s_PropertyBlock);
                
                s_PropertyBlock.Clear();
            }
        }
        
        NewGSRenderPass m_Pass;
        bool m_HasCamera;
        
        public Shader blitShader;

        public override void Create()
        {
           m_Pass = new NewGSRenderPass
            {
                renderPassEvent = RenderPassEvent.BeforeRenderingTransparents,
                blitMaterial = CoreUtils.CreateEngineMaterial(blitShader)
            };
        }

        public override void OnCameraPreCull(ScriptableRenderer renderer, in CameraData cameraData)
        {
            m_HasCamera = false;
            var system = NewGaussianSplatRenderSystem.instance;
            if (!system.GatherSplatsForCamera(cameraData.camera))
                return;

            m_HasCamera = true;
        }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            if (!m_HasCamera)
                return;
            renderer.EnqueuePass(m_Pass);
        }

        protected override void Dispose(bool disposing)
        {
            m_Pass.CleanUp();
            m_Pass = null;
        }
    }
}

#endif // #if GS_ENABLE_URP
