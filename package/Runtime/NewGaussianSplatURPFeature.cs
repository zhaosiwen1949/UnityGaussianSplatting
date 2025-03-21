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

namespace GaussianSplatting.Runtime
{
    // Note: I have no idea what is the purpose of ScriptableRendererFeature vs ScriptableRenderPass, which one of those
    // is supposed to do resource management vs logic, etc. etc. Code below "seems to work" but I'm just fumbling along,
    // without understanding any of it.
    //
    // ReSharper disable once InconsistentNaming
    class NewGaussianSplatURPFeature : ScriptableRendererFeature
    {
        class SharedHandle
        {
            public BufferHandle m_GeomStateBuffer;
        }
        
        class PreProcessRenderPass : ScriptableRenderPass
        {
            const string ProfilerTag = "GaussianSplatPreProcess";
            static readonly ProfilingSampler s_profilingSampler = new(ProfilerTag);
            internal static readonly ProfilerMarker s_ProfSampler = new(ProfilerCategory.Render, "GSPass.PreProcess", MarkerFlags.SampleGPU);
            
            private SharedHandle m_sharedHandle;

            public void SetShareHandle(SharedHandle sharedHandle)
            {
                m_sharedHandle = sharedHandle;
            }
            
            class PassData
            {
                internal UniversalCameraData CameraData;
                internal BufferHandle GeomStateData;
                internal NewGaussianSplatRenderer GSRenderer;
            }
            
            public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
            {
                // using var builder = renderGraph.AddUnsafePass(ProfilerTag, out PassData passData);
                // var cameraData = frameData.Get<UniversalCameraData>();
                //
                // BufferDesc geomStateDesc = new BufferDesc
                // {
                //     count = NewGaussianSplatRenderSystem.instance.GetFirstRenderer().splatCount,
                //     stride = 16 * 4,
                //     target = GraphicsBuffer.Target.Structured,
                //     name = "GSStateBuffer"
                // };
                // m_sharedHandle.m_GeomStateBuffer = renderGraph.CreateBuffer(geomStateDesc);
                //
                // passData.CameraData = cameraData;
                // passData.GeomStateData = m_sharedHandle.m_GeomStateBuffer;
                // passData.GSRenderer = NewGaussianSplatRenderSystem.instance.GetFirstRenderer();
                //
                // builder.UseBuffer(m_sharedHandle.m_GeomStateBuffer, AccessFlags.Write);
                // builder.AllowPassCulling(false);
                // builder.SetRenderFunc(static (PassData data, UnsafeGraphContext context) =>
                // {
                //     var commandBuffer = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
                //     using var _ = new ProfilingScope(commandBuffer, s_profilingSampler);
                //     commandBuffer.BeginSample(s_ProfSampler);
                //     data.GSRenderer.PreProcessViewData(commandBuffer, data.CameraData.camera, data.GeomStateData);
                //     commandBuffer.EndSample(s_ProfSampler);
                // });
            }
        }
        
        class NewGSRenderPass : ScriptableRenderPass
        {
            const string GaussianSplatRTName = "_GaussianSplatRT";

            const string ProfilerTag = "GaussianSplatRenderGraph";
            static readonly ProfilingSampler s_profilingSampler = new(ProfilerTag);
            static readonly int s_gaussianSplatRT = Shader.PropertyToID(GaussianSplatRTName);
            
            private SharedHandle m_sharedHandle;

            public void SetShareHandle(SharedHandle sharedHandle)
            {
                m_sharedHandle = sharedHandle;
            }
            
            class PassData
            {
                internal UniversalCameraData CameraData;
                internal TextureHandle SourceTexture;
                internal TextureHandle SourceDepth;
                internal TextureHandle GaussianSplatRT;
            }

            public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
            {
                using var builder = renderGraph.AddUnsafePass(ProfilerTag, out PassData passData);

                var cameraData = frameData.Get<UniversalCameraData>();
                var resourceData = frameData.Get<UniversalResourceData>();

                RenderTextureDescriptor rtDesc = cameraData.cameraTargetDescriptor;
                rtDesc.depthBufferBits = 0;
                rtDesc.msaaSamples = 1;
                // rtDesc.graphicsFormat = GraphicsFormat.R16G16B16A16_SFloat;
                rtDesc.graphicsFormat = GraphicsFormat.R32G32B32A32_SFloat;
                rtDesc.enableRandomWrite = true;
                var textureHandle = UniversalRenderer.CreateRenderGraphTexture(renderGraph, rtDesc, GaussianSplatRTName, true);

                passData.CameraData = cameraData;
                passData.SourceTexture = resourceData.activeColorTexture;
                passData.SourceDepth = resourceData.activeDepthTexture;
                passData.GaussianSplatRT = textureHandle;

                builder.UseTexture(resourceData.activeColorTexture, AccessFlags.ReadWrite);
                builder.UseTexture(resourceData.activeDepthTexture);
                builder.UseTexture(textureHandle, AccessFlags.Write);
                builder.AllowPassCulling(false);
                builder.SetRenderFunc(static (PassData data, UnsafeGraphContext context) =>
                {
                    var commandBuffer = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
                    using var _ = new ProfilingScope(commandBuffer, s_profilingSampler);
                    // commandBuffer.SetGlobalTexture(s_gaussianSplatRT, data.GaussianSplatRT);
                    // CoreUtils.SetRenderTarget(commandBuffer, data.GaussianSplatRT, data.SourceDepth, ClearFlag.Color, Color.clear);
                    // Material matComposite = NewGaussianSplatRenderSystem.instance.SortAndRenderSplats(data.CameraData.camera, commandBuffer);
                    // commandBuffer.BeginSample(NewGaussianSplatRenderSystem.s_ProfCompose);
                    // Blitter.BlitCameraTexture(commandBuffer, data.GaussianSplatRT, data.SourceTexture, matComposite, 0);
                    // commandBuffer.EndSample(NewGaussianSplatRenderSystem.s_ProfCompose);
                    
                    // Tile 渲染方法
                    NewGaussianSplatRenderSystem.instance.TileRenderSplats(data.CameraData.camera, commandBuffer, data.GaussianSplatRT);
                    commandBuffer.BeginSample(NewGaussianSplatRenderSystem.s_ProfCompose);
                    Blitter.BlitCameraTexture(commandBuffer, data.GaussianSplatRT, data.SourceTexture);
                    commandBuffer.EndSample(NewGaussianSplatRenderSystem.s_ProfCompose);
                });
            }
        }
        
        PreProcessRenderPass m_PreProcessRenderPass;
        NewGSRenderPass m_Pass;
        bool m_HasCamera;

        SharedHandle m_sharedHandle;

        public override void Create()
        {
            m_sharedHandle = new SharedHandle();
            
            // m_PreProcessRenderPass = new PreProcessRenderPass
            // {
            //     renderPassEvent = RenderPassEvent.BeforeRenderingTransparents
            // };
            // m_PreProcessRenderPass.SetShareHandle(m_sharedHandle);
            
            m_Pass = new NewGSRenderPass
            {
                renderPassEvent = RenderPassEvent.BeforeRenderingTransparents
            };
            m_Pass.SetShareHandle(m_sharedHandle);
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
            // renderer.EnqueuePass(m_PreProcessRenderPass);
            renderer.EnqueuePass(m_Pass);
        }

        protected override void Dispose(bool disposing)
        {
            m_Pass = null;
        }
    }
}

#endif // #if GS_ENABLE_URP
