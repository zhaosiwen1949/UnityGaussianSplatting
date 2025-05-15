// SPDX-License-Identifier: MIT

using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.XR;

namespace GaussianSplatting.Runtime
{
    class NewGaussianSplatRenderSystem
    {
        // ReSharper disable MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        internal static readonly ProfilerMarker s_ProfDraw = new(ProfilerCategory.Render, "GaussianSplat.Draw", MarkerFlags.SampleGPU);
        internal static readonly ProfilerMarker s_ProfCompose = new(ProfilerCategory.Render, "GaussianSplat.Compose", MarkerFlags.SampleGPU);
        internal static readonly ProfilerMarker s_ProfCalcView = new(ProfilerCategory.Render, "GaussianSplat.CalcView", MarkerFlags.SampleGPU);
        internal static readonly ProfilerMarker s_ProfPreProcess = new(ProfilerCategory.Render, "GaussianSplat.PreProcess", MarkerFlags.SampleGPU);
        internal static readonly ProfilerMarker s_ProfRadixSort = new(ProfilerCategory.Render, "GaussianSplat.RadixSort", MarkerFlags.SampleGPU);
        // ReSharper restore MemberCanBePrivate.Global

        public static NewGaussianSplatRenderSystem instance => ms_Instance ??= new NewGaussianSplatRenderSystem();
        static NewGaussianSplatRenderSystem ms_Instance;

        readonly Dictionary<NewGaussianSplatRenderer, MaterialPropertyBlock> m_Splats = new();
        readonly HashSet<Camera> m_CameraCommandBuffersDone = new();
        readonly List<(NewGaussianSplatRenderer, MaterialPropertyBlock)> m_ActiveSplats = new();

        CommandBuffer m_CommandBuffer;

        public NewGaussianSplatRenderer GetFirstRenderer()
        {
            return m_ActiveSplats.Count > 0 ? m_ActiveSplats[0].Item1 : null;
        }
        
        public void RegisterSplat(NewGaussianSplatRenderer r)
        {
            m_Splats.Add(r, new MaterialPropertyBlock());
        }

        public void UnregisterSplat(NewGaussianSplatRenderer r)
        {
            if (!m_Splats.ContainsKey(r))
                return;
            m_Splats.Remove(r);
            if (m_Splats.Count == 0)
            {
                if (m_CameraCommandBuffersDone != null)
                {
                    if (m_CommandBuffer != null)
                    {
                        foreach (var cam in m_CameraCommandBuffersDone)
                        {
                            if (cam)
                                cam.RemoveCommandBuffer(CameraEvent.BeforeForwardAlpha, m_CommandBuffer);
                        }
                    }
                    m_CameraCommandBuffersDone.Clear();
                }

                m_ActiveSplats.Clear();
                m_CommandBuffer?.Dispose();
                m_CommandBuffer = null;
            }
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        public bool GatherSplatsForCamera(Camera cam)
        {
            if (cam.cameraType == CameraType.Preview)
                return false;
            // gather all active & valid splat objects
            m_ActiveSplats.Clear();
            foreach (var kvp in m_Splats)
            {
                var gs = kvp.Key;
                if (gs == null || !gs.isActiveAndEnabled || !gs.HasValidAsset || !gs.HasValidRenderSetup)
                    continue;
                m_ActiveSplats.Add((kvp.Key, kvp.Value));
            }
            if (m_ActiveSplats.Count == 0)
                return false;

            // sort them by order and depth from camera
            var camTr = cam.transform;
            m_ActiveSplats.Sort((a, b) =>
            {
                var orderA = a.Item1.m_RenderOrder;
                var orderB = b.Item1.m_RenderOrder;
                if (orderA != orderB)
                    return orderB.CompareTo(orderA);
                var trA = a.Item1.transform;
                var trB = b.Item1.transform;
                var posA = camTr.InverseTransformPoint(trA.position);
                var posB = camTr.InverseTransformPoint(trB.position);
                return posA.z.CompareTo(posB.z);
            });

            return true;
        }

        public void TileRenderSplats(
            Camera cam,
            CommandBuffer cmb,
            TextureHandle gsRenderTexture,
            TextureHandle preDepthTexture,
            TextureHandle currentDepthTexture,
            Matrix4x4 preViewProjectionMatrix
        )
        {
            if (m_ActiveSplats.Count <= 0) return;
            
            var kvp = m_ActiveSplats[0];
            var gs = kvp.Item1;
            
            // PreProcess
            cmb.BeginSample(s_ProfPreProcess);
            gs.PreProcessViewData(cmb, cam, preDepthTexture, preViewProjectionMatrix);
            cmb.EndSample(s_ProfPreProcess);
                
            // RadixSort
            cmb.BeginSample(s_ProfRadixSort);
            gs.RadixSortPoints(cmb, cam);
            cmb.EndSample(s_ProfRadixSort);
            
            // TileRender
            cmb.BeginSample(s_ProfDraw);
            gs.RenderViewData(cmb, cam, gsRenderTexture, currentDepthTexture);
            cmb.EndSample(s_ProfDraw);
        }
    }
}