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
            const string CurrentGaussianSplatDepthName = "_CurrentGaussianSplatDepth";
            const string PreGaussianSplatDepthName = "_PreGaussianSplatDepth";

            const string ProfilerTag = "GaussianSplatRenderGraph";
            static readonly ProfilingSampler s_profilingSampler = new(ProfilerTag);
            static readonly int s_gaussianSplatRT = Shader.PropertyToID(GaussianSplatRTName);
            
            private RTHandle m_preDepthTexture;
            private RTHandle m_currentDepthTexture;
            private Matrix4x4 m_preViewProjectionMatrix = Matrix4x4.identity;
            private Matrix4x4 m_preLeftViewProjectionMatrix = Matrix4x4.identity;
            private Matrix4x4 m_preRightViewProjectionMatrix = Matrix4x4.identity;
            
            class PassData
            {
                internal UniversalCameraData CameraData;
                internal TextureHandle SourceTexture;
                internal TextureHandle SourceDepth;
                internal TextureHandle GaussianSplatRT;
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

                RenderTextureDescriptor rtDesc = cameraData.cameraTargetDescriptor;
                rtDesc.depthBufferBits = 0;
                rtDesc.msaaSamples = 1;
                rtDesc.autoGenerateMips = false;
                rtDesc.graphicsFormat = GraphicsFormat.R16G16B16A16_SFloat;
                rtDesc.enableRandomWrite = true;
                var textureHandle = UniversalRenderer.CreateRenderGraphTexture(renderGraph, rtDesc, GaussianSplatRTName, true);
                
                // 初始化创建 PreDepthTexture 和 CurrentDepthTexture
                if (m_preDepthTexture == null)
                {
                    RenderTextureDescriptor depthDesc = new RenderTextureDescriptor(rtDesc.width, rtDesc.height);
                    depthDesc.depthBufferBits = 0;
                    depthDesc.msaaSamples = 1;
                    depthDesc.autoGenerateMips = false;
                    depthDesc.graphicsFormat = XRSettings.stereoRenderingMode == XRSettings.StereoRenderingMode.SinglePassInstanced ? GraphicsFormat.R16G16_SFloat : GraphicsFormat.R16_SFloat;
                    depthDesc.enableRandomWrite = true;
                    RenderingUtils.ReAllocateIfNeeded(ref m_preDepthTexture, depthDesc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: PreGaussianSplatDepthName );
                }
                
                if (m_currentDepthTexture == null)
                {
                    RenderTextureDescriptor depthDesc = new RenderTextureDescriptor(rtDesc.width, rtDesc.height);
                    depthDesc.depthBufferBits = 0;
                    depthDesc.msaaSamples = 1;
                    depthDesc.autoGenerateMips = false;
                    depthDesc.graphicsFormat = XRSettings.stereoRenderingMode == XRSettings.StereoRenderingMode.SinglePassInstanced ? GraphicsFormat.R16G16_SFloat : GraphicsFormat.R16_SFloat;
                    depthDesc.enableRandomWrite = true;
                    RenderingUtils.ReAllocateIfNeeded(ref m_currentDepthTexture, depthDesc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: CurrentGaussianSplatDepthName );
                }
                
                // 交换 PreDepthTexture 和 CurrentDepthTexture
                (m_preDepthTexture, m_currentDepthTexture) = (m_currentDepthTexture, m_preDepthTexture);

                // 引入 PreDepthTexture 和 CurrentDepthTexture
                TextureHandle preDepthTextureHandle = renderGraph.ImportTexture(m_preDepthTexture);
                TextureHandle currentDepthTextureHandle = renderGraph.ImportTexture(m_currentDepthTexture);

                passData.CameraData = cameraData;
                passData.SourceTexture = resourceData.activeColorTexture;
                passData.SourceDepth = resourceData.activeDepthTexture;
                passData.GaussianSplatRT = textureHandle;
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
                builder.UseTexture(preDepthTextureHandle);
                builder.UseTexture(currentDepthTextureHandle, AccessFlags.Write);
                
                builder.AllowPassCulling(false);
                builder.SetRenderFunc(static (PassData data, UnsafeGraphContext context) =>
                {
                    var commandBuffer = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
                    using var _ = new ProfilingScope(commandBuffer, s_profilingSampler);
                    
                    // Tile 渲染方法
                    NewGaussianSplatRenderSystem.instance.TileRenderSplats(
                        data.CameraData.camera,
                        commandBuffer,
                        data.GaussianSplatRT,
                        data.PreGaussianSplatDepth,
                        data.CurrentGaussianSplatDepth,
                        data.PreLeftViewProjectionMatrix,
                        data.PreRightViewProjectionMatrix
                    );
                    commandBuffer.BeginSample(NewGaussianSplatRenderSystem.s_ProfCompose);
                    Blitter.BlitCameraTexture(commandBuffer, data.GaussianSplatRT, data.SourceTexture);
                    commandBuffer.EndSample(NewGaussianSplatRenderSystem.s_ProfCompose);
                });
            }

            public void CleanUp()
            {
                m_preDepthTexture?.Release();
                m_currentDepthTexture?.Release();
            }
        }
        
        NewGSRenderPass m_Pass;
        bool m_HasCamera;

        public override void Create()
        {
           m_Pass = new NewGSRenderPass
            {
                renderPassEvent = RenderPassEvent.BeforeRenderingTransparents
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
