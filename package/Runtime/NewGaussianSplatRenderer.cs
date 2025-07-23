// SPDX-License-Identifier: MIT
using System;
using GPUInt64Sorting.Runtime;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.XR;

namespace GaussianSplatting.Runtime
{
    struct Int4Data
    {
        private int a, b, c, d;
        public int GetElement(int index)
        {
            switch (index)
            {
                case 0:
                    return a;
                case 1:
                    return b;
                case 2:
                    return c;
                case 3:
                    return d;
                default:
                    return 0;
            }
        }
    }
    
    struct Int2Data
    {
        private int a, b;
        public int GetElement(int index)
        {
            switch (index)
            {
                case 0:
                    return a;
                case 1:
                    return b;
                default:
                    return 0;
            }
        }
    }

    struct PlaneData
    {
        public Vector3 normal;
        public float distance;
        public Vector3 center;
        public float height;
        public float width;
        public Vector3 heightDir;
        public Vector3 widthDir;
    }

    [ExecuteInEditMode]
    public class NewGaussianSplatRenderer : MonoBehaviour
    {
        static int MAX_DISPATCH_GROUP = 65535; 
        public GaussianSplatAsset m_Asset;

        [Tooltip("Rendering order compared to other splats. Within same order splats are sorted by distance. Higher order splats render 'on top of' lower order splats.")]
        public int m_RenderOrder;
        [Range(0.1f, 2.0f)] [Tooltip("Additional scaling factor for the splats")]
        public float m_SplatScale = 1.0f;
        [Range(0.05f, 20.0f)]
        [Tooltip("Additional scaling factor for opacity")]
        public float m_OpacityScale = 1.0f;
        [Range(0, 3)] [Tooltip("Spherical Harmonics order to use")]
        public int m_SHOrder = 3;
        [Tooltip("Show only Spherical Harmonics contribution, using gray color")]
        public bool m_SHOnly;
        [Range(1,30)] [Tooltip("Sort splats only every N frames")]
        public int m_SortNthFrame = 1;
        [Tooltip("Occlusion Plane list")] public GameObject[] m_OcclusionPlanes;

        [Range(1, 10)] public int m_TileScale = 2;
        [Range(0.5f, 1.0f)] public float m_WidthScale = 0.75f;
        [Range(0.5f, 1.0f)] public float m_HeightScale = 0.75f;
        [Range(0.1f, 1.0f)] public float m_TextureScale = 0.5f;
        [Range(0.05f, 1.0f)] public float m_DepthCullingThreshold = 0.4f;
        [Range(0.0f, 1.0f)] public float m_Sharpness = 0.92f;
        

        public GaussianCutout[] m_Cutouts;

       [Tooltip("Gaussian splatting compute shader")]
        public ComputeShader m_CSSplatUtilities;

        int m_SplatCount; // initially same as asset splat count, but editing can change this
        private int m_TileRenderCount;
        GraphicsBuffer m_GpuSortDistances;
        internal GraphicsBuffer m_GpuSortKeys;
        GraphicsBuffer m_GpuPosData;
        GraphicsBuffer m_GpuOtherData;
        GraphicsBuffer m_GpuSHData;
        Texture m_GpuColorData;
        internal GraphicsBuffer m_GpuChunks;
        internal bool m_GpuChunksValid;
        
        // new tile-renderer needed buffer
        // OcclusionCullingPlane
        internal GraphicsBuffer m_PlaneDataBuffer;
        
        // VisibleCounts
        internal GraphicsBuffer m_NumArgs_left;
        internal GraphicsBuffer m_NumArgs_right;
        
        // GeometryState
        internal GraphicsBuffer m_GeomState_left_data;
        internal GraphicsBuffer m_GeomState_right_data;

        // BinningState
        internal GraphicsBuffer m_BinState_left_point_list_keys;
        internal GraphicsBuffer m_BinState_left_point_list_values;
        internal GraphicsBuffer m_BinState_right_point_list_keys;
        internal GraphicsBuffer m_BinState_right_point_list_values;

        // ImageState
        internal GraphicsBuffer m_ImageState_left_ranges;
        internal GraphicsBuffer m_ImageState_right_ranges;
        
        // Radix Sorter
        private NewDeviceRadixSort m_RadixSorter_left;
        private GraphicsBuffer m_AltKey_left;
        private GraphicsBuffer m_AltPayload_left;
        private GraphicsBuffer m_GlobalHist_left;
        private GraphicsBuffer m_PassHist_left;
        
        private NewDeviceRadixSort m_RadixSorter_right;
        private GraphicsBuffer m_AltKey_right;
        private GraphicsBuffer m_AltPayload_right;
        private GraphicsBuffer m_GlobalHist_right;
        private GraphicsBuffer m_PassHist_right;

        private PlaneData[] m_planeDataList = new PlaneData[10];
        
        // Shader Keyword
        private LocalKeyword m_SinglePassStereoKeyWord;
        private LocalKeyword m_SinglePassSingleKeyWord;
        
        GaussianSplatAsset m_PrevAsset;
        Hash128 m_PrevHash;
        bool m_Registered;

        private int m_PreTileX, m_PreTileY;
        
        internal static class Props
        {
            public static readonly int SplatPos = Shader.PropertyToID("_SplatPos");
            public static readonly int SplatOther = Shader.PropertyToID("_SplatOther");
            public static readonly int SplatSH = Shader.PropertyToID("_SplatSH");
            public static readonly int SplatColor = Shader.PropertyToID("_SplatColor");
            public static readonly int SplatSelectedBits = Shader.PropertyToID("_SplatSelectedBits");
            public static readonly int SplatDeletedBits = Shader.PropertyToID("_SplatDeletedBits");
            public static readonly int SplatBitsValid = Shader.PropertyToID("_SplatBitsValid");
            public static readonly int SplatFormat = Shader.PropertyToID("_SplatFormat");
            public static readonly int SplatChunks = Shader.PropertyToID("_SplatChunks");
            public static readonly int SplatChunkCount = Shader.PropertyToID("_SplatChunkCount");
            public static readonly int SplatScale = Shader.PropertyToID("_SplatScale");
            public static readonly int SplatOpacityScale = Shader.PropertyToID("_SplatOpacityScale");
            public static readonly int SplatSize = Shader.PropertyToID("_SplatSize");
            public static readonly int SplatCount = Shader.PropertyToID("_SplatCount");
            public static readonly int SHOrder = Shader.PropertyToID("_SHOrder");
            public static readonly int SHOnly = Shader.PropertyToID("_SHOnly");
            
            public static readonly int LeftNumArgs = Shader.PropertyToID("_LeftNumArgs");
            public static readonly int RightNumArgs = Shader.PropertyToID("_RightNumArgs");
            public static readonly int GeomLeftData = Shader.PropertyToID("_GeomLeftData");
            public static readonly int GeomRightData = Shader.PropertyToID("_GeomRightData");
            public static readonly int BinLeftPointListKey = Shader.PropertyToID("_BinLeftPointListKey");
            public static readonly int BinLeftPointListValue = Shader.PropertyToID("_BinLeftPointListValue");
            public static readonly int BinRightPointListKey = Shader.PropertyToID("_BinRightPointListKey");
            public static readonly int BinRightPointListValue = Shader.PropertyToID("_BinRightPointListValue");
            public static readonly int ImageLeftRange = Shader.PropertyToID("_ImageLeftRange");
            public static readonly int ImageRightRange = Shader.PropertyToID("_ImageRightRange");
            public static readonly int ImageRange = Shader.PropertyToID("_ImageRange");
            
            public static readonly int RO_PlaneData = Shader.PropertyToID("_RO_PlaneData");
            public static readonly int RO_NumArgs = Shader.PropertyToID("_RO_NumArgs");
            public static readonly int RO_GeomLeftData = Shader.PropertyToID("_RO_GeomLeftData");
            public static readonly int RO_GeomRightData = Shader.PropertyToID("_RO_GeomRightData");
            public static readonly int RO_BinPointListKey = Shader.PropertyToID("_RO_BinPointListKey");
            public static readonly int RO_ImageLeftRange = Shader.PropertyToID("_RO_ImageLeftRange");
            public static readonly int RO_ImageRightRange = Shader.PropertyToID("_RO_ImageRightRange");
            public static readonly int RO_BinLeftPointListValue = Shader.PropertyToID("_RO_BinLeftPointListValue");
            public static readonly int RO_BinRightPointListValue = Shader.PropertyToID("_RO_BinRightPointListValue");
            
            public static readonly int NumOcclusionPlane = Shader.PropertyToID("_NumOcclusionPlane");
            public static readonly int TileConfig = Shader.PropertyToID("_TileConfig");
            public static readonly int TileScale = Shader.PropertyToID("_TileScale");
            public static readonly int DepthCullingThreshold = Shader.PropertyToID("_DepthCullingThreshold");
            public static readonly int GSRenderTexture = Shader.PropertyToID("_GSRenderTexture");
            public static readonly int GSPreDepthTexture = Shader.PropertyToID("_GSPreDepthTexture");
            public static readonly int GSDepthTexture = Shader.PropertyToID("_GSDepthTexture");
            
            public static readonly int DisplayIndex = Shader.PropertyToID("_DisplayIndex");
            public static readonly int DisplayChunks = Shader.PropertyToID("_DisplayChunks");
            public static readonly int GaussianSplatRT = Shader.PropertyToID("_GaussianSplatRT");
            public static readonly int SplatSortKeys = Shader.PropertyToID("_SplatSortKeys");
            public static readonly int SplatSortDistances = Shader.PropertyToID("_SplatSortDistances");
            public static readonly int SrcBuffer = Shader.PropertyToID("_SrcBuffer");
            public static readonly int DstBuffer = Shader.PropertyToID("_DstBuffer");
            public static readonly int BufferSize = Shader.PropertyToID("_BufferSize");
            public static readonly int MatrixMV = Shader.PropertyToID("_MatrixMV");
            public static readonly int MatrixPreLeftVP = Shader.PropertyToID("_MatrixPreLeftVP");
            public static readonly int MatrixPreRightVP = Shader.PropertyToID("_MatrixPreRightVP");
            public static readonly int MatrixObjectToWorld = Shader.PropertyToID("_MatrixObjectToWorld");
            public static readonly int MatrixWorldToObject = Shader.PropertyToID("_MatrixWorldToObject");
            public static readonly int MatrixLV = Shader.PropertyToID("_MatrixLV");
            public static readonly int MatrixRV = Shader.PropertyToID("_MatrixRV");
            public static readonly int MatrixLP = Shader.PropertyToID("_MatrixLP");
            public static readonly int MatrixRP = Shader.PropertyToID("_MatrixRP");
            public static readonly int MatrixRPRVLV = Shader.PropertyToID("_MatrixRPRVLV");
            public static readonly int Focal = Shader.PropertyToID("_Focal");
            public static readonly int VecScreenParams = Shader.PropertyToID("_VecScreenParams");
            public static readonly int VecWorldSpaceCameraPos = Shader.PropertyToID("_VecWorldSpaceCameraPos");
            public static readonly int CameraTargetTexture = Shader.PropertyToID("_CameraTargetTexture");
            public static readonly int SelectionCenter = Shader.PropertyToID("_SelectionCenter");
            public static readonly int SelectionDelta = Shader.PropertyToID("_SelectionDelta");
            public static readonly int SelectionDeltaRot = Shader.PropertyToID("_SelectionDeltaRot");
            public static readonly int SplatCutoutsCount = Shader.PropertyToID("_SplatCutoutsCount");
            public static readonly int SplatCutouts = Shader.PropertyToID("_SplatCutouts");
            public static readonly int SelectionMode = Shader.PropertyToID("_SelectionMode");
            public static readonly int SplatPosMouseDown = Shader.PropertyToID("_SplatPosMouseDown");
            public static readonly int SplatOtherMouseDown = Shader.PropertyToID("_SplatOtherMouseDown");
        }

        public GaussianSplatAsset asset => m_Asset;

        enum KernelIndices
        {
            PreProcessViewData,
            IdentifyTileRanges,
            InitImageRanges,
            InitNumArgs,
            InitSortArgs,
            RenderViewData,
        }

        public bool HasValidAsset =>
            m_Asset != null &&
            m_Asset.splatCount > 0 &&
            m_Asset.formatVersion == GaussianSplatAsset.kCurrentVersion &&
            m_Asset.posData != null &&
            m_Asset.otherData != null &&
            m_Asset.shData != null &&
            m_Asset.colorData != null;
        public bool HasValidRenderSetup => m_GpuPosData != null && m_GpuOtherData != null && m_GpuChunks != null;

        private bool m_IsSinglePass =>
            XRSettings.stereoRenderingMode == XRSettings.StereoRenderingMode.SinglePassInstanced;

        void CreateResourcesForAsset()
        {
            if (!HasValidAsset || !resourcesAreSetUp)
                return;
            
            // 初始化 ShaderKeywords
            m_SinglePassStereoKeyWord = new LocalKeyword(m_CSSplatUtilities, "SINGLE_PASS_STEREO");
            m_SinglePassSingleKeyWord = new LocalKeyword(m_CSSplatUtilities, "SINGLE_PASS_SINGLE");

            m_SplatCount = asset.splatCount;
            m_TileRenderCount = 5 * asset.splatCount;
            m_GpuPosData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, (int) (asset.posData.dataSize / 4), 4) { name = "GaussianPosData" };
            m_GpuPosData.SetData(asset.posData.GetData<uint>());
            m_GpuOtherData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, (int) (asset.otherData.dataSize / 4), 4) { name = "GaussianOtherData" };
            m_GpuOtherData.SetData(asset.otherData.GetData<uint>());
            m_GpuSHData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, (int) (asset.shData.dataSize / 4), 4) { name = "GaussianSHData" };
            m_GpuSHData.SetData(asset.shData.GetData<uint>());
            var (texWidth, texHeight) = GaussianSplatAsset.CalcTextureSize(asset.splatCount);
            var texFormat = GaussianSplatAsset.ColorFormatToGraphics(asset.colorFormat);
            var tex = new Texture2D(texWidth, texHeight, texFormat, TextureCreationFlags.DontInitializePixels | TextureCreationFlags.IgnoreMipmapLimit | TextureCreationFlags.DontUploadUponCreate) { name = "GaussianColorData" };
            tex.SetPixelData(asset.colorData.GetData<byte>(), 0);
            tex.Apply(false, true);
            m_GpuColorData = tex;
            if (asset.chunkData != null && asset.chunkData.dataSize != 0)
            {
                m_GpuChunks = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                    (int) (asset.chunkData.dataSize / UnsafeUtility.SizeOf<GaussianSplatAsset.ChunkInfo>()),
                    UnsafeUtility.SizeOf<GaussianSplatAsset.ChunkInfo>()) {name = "GaussianChunkData"};
                m_GpuChunks.SetData(asset.chunkData.GetData<GaussianSplatAsset.ChunkInfo>());
                m_GpuChunksValid = true;
            }
            else
            {
                // just a dummy chunk buffer
                m_GpuChunks = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1,
                    UnsafeUtility.SizeOf<GaussianSplatAsset.ChunkInfo>()) {name = "GaussianChunkData"};
                m_GpuChunksValid = false;
            }
            
            m_PlaneDataBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_planeDataList.Length, UnsafeUtility.SizeOf<PlaneData>());
            UpdateOcclusionPlane();
            
            // 初始化 NumArgs
            m_NumArgs_left = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 8, 4) {name = "NumArgsLeft"};
            m_NumArgs_right = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 8, 4) {name = "NumArgsRight"};
            // uint[] zero = new uint[8];
            // m_NumArgs.SetData(zero);
            
            // 初始化 GeometryState
            m_GeomState_left_data =
                new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_SplatCount, 8 * 4)
                    { name = "GeomStateLeftData" };
            m_GeomState_right_data =
                new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_SplatCount, 8 * 4)
                    { name = "GeomStateRightData" };
            
            // BinState 中深度排序的部分，和点云点数保持相同
            m_BinState_left_point_list_keys = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_TileRenderCount, 8)
                { name = "BinStateLeftKeyData" };
            m_BinState_left_point_list_values = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_TileRenderCount, 4)
                { name = "BinStateLeftValueData" };
            m_BinState_right_point_list_keys = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_TileRenderCount, 8)
                { name = "BinStateRightKeyData" };
            m_BinState_right_point_list_values = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_TileRenderCount, 4)
                { name = "BinStateRightValueData" };
            
            InitRadixSortBuffers(m_TileRenderCount);

        }

        public void EnsureImageState(Camera cam, bool isStereo)
        {
            GetTileConfig(m_CSSplatUtilities, cam, out var tile_x, out var tile_y, out _, out _);
            if (m_PreTileX != tile_x || m_PreTileY != tile_y || m_ImageState_left_ranges == null || m_ImageState_right_ranges == null)
            {
                // 更新之前先销毁原来的数据
                DisposeBuffer(ref m_ImageState_left_ranges);
                DisposeBuffer(ref m_ImageState_right_ranges);

                // 设置 ImageState
                m_ImageState_left_ranges = new GraphicsBuffer(GraphicsBuffer.Target.Structured, tile_x * tile_y, 4 * 2)
                    { name = "ImageStateLeftRangesData" };
                m_ImageState_right_ranges = new GraphicsBuffer(GraphicsBuffer.Target.Structured, tile_x * tile_y, 4 * 2)
                    { name = "ImageStateRightRangesData" };
                    
                // 把 m_ImageState_left_ranges 置为 0
                {
                    int tile_count = tile_x * tile_y;
                    m_CSSplatUtilities.SetBuffer((int)KernelIndices.InitImageRanges, Props.ImageLeftRange, m_ImageState_left_ranges);
                    if (m_IsSinglePass && isStereo)
                    {
                        m_CSSplatUtilities.SetBuffer((int)KernelIndices.InitImageRanges, Props.ImageRightRange, m_ImageState_right_ranges);
                    }
                    m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.InitImageRanges, out uint gsX,
                        out _, out _);
                    GetDispatchGroupNum(tile_count, (int)gsX, out int countX, out int countY);
                    m_CSSplatUtilities.Dispatch( (int)KernelIndices.InitImageRanges,
                        countX, countY, 1);
                }
                
                m_PreTileX = tile_x;
                m_PreTileY = tile_y;
            }
        }
        
        void InitRadixSortBuffers(int tileRenderCount)
        {
            m_RadixSorter_left = new NewDeviceRadixSort(
                m_CSSplatUtilities,
                tileRenderCount,
                ref m_AltKey_left,
                ref m_AltPayload_left,
                ref m_GlobalHist_left,
                ref m_PassHist_left);
            
            m_RadixSorter_right = new NewDeviceRadixSort(
                m_CSSplatUtilities,
                tileRenderCount,
                ref m_AltKey_right,
                ref m_AltPayload_right,
                ref m_GlobalHist_right,
                ref m_PassHist_right);
        }

        bool resourcesAreSetUp => m_CSSplatUtilities != null && SystemInfo.supportsComputeShaders;

        public void EnsureRegister()
        {
            if (!m_Registered && resourcesAreSetUp)
            {
                NewGaussianSplatRenderSystem.instance.RegisterSplat(this);
                m_Registered = true;
            }
        }

        public void OnEnable()
        {
            if (!resourcesAreSetUp)
                return;
            
            EnsureRegister();

            CreateResourcesForAsset();
        }

        void SetAssetDataOnCS(CommandBuffer cmb, KernelIndices kernel)
        {
            ComputeShader cs = m_CSSplatUtilities;
            int kernelIndex = (int) kernel;
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatPos, m_GpuPosData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatChunks, m_GpuChunks);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatOther, m_GpuOtherData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatSH, m_GpuSHData);
            cmb.SetComputeTextureParam(cs, kernelIndex, Props.SplatColor, m_GpuColorData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatSelectedBits, m_GpuPosData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatDeletedBits, m_GpuPosData);

            cmb.SetComputeIntParam(cs, Props.SplatBitsValid,  0);
            uint format = (uint)m_Asset.posFormat | ((uint)m_Asset.scaleFormat << 8) | ((uint)m_Asset.shFormat << 16);
            cmb.SetComputeIntParam(cs, Props.SplatFormat, (int)format);
            cmb.SetComputeIntParam(cs, Props.SplatCount, m_SplatCount);
            cmb.SetComputeIntParam(cs, Props.SplatChunkCount, m_GpuChunksValid ? m_GpuChunks.count : 0);

            cmb.SetComputeIntParam(cs, Props.SplatCutoutsCount, m_Cutouts?.Length ?? 0);
        }

        static void DisposeBuffer(ref GraphicsBuffer buf)
        {
            buf?.Dispose();
            buf = null;
        }

        void DisposeResourcesForAsset()
        {
            DestroyImmediate(m_GpuColorData);

            DisposeBuffer(ref m_GpuPosData);
            DisposeBuffer(ref m_GpuOtherData);
            DisposeBuffer(ref m_GpuSHData);
            DisposeBuffer(ref m_GpuChunks);
            
            DisposeBuffer(ref m_AltKey_left);
            DisposeBuffer(ref m_AltPayload_left);
            DisposeBuffer(ref m_GlobalHist_left);
            DisposeBuffer(ref m_PassHist_left);
            DisposeBuffer(ref m_AltKey_right);
            DisposeBuffer(ref m_AltPayload_right);
            DisposeBuffer(ref m_GlobalHist_right);
            DisposeBuffer(ref m_PassHist_right);
            
            DisposeBuffer(ref m_PlaneDataBuffer);
            DisposeBuffer(ref m_NumArgs_left);
            DisposeBuffer(ref m_NumArgs_right);
            DisposeBuffer(ref m_GeomState_left_data);
            DisposeBuffer(ref m_GeomState_right_data);
            
            DisposeBuffer(ref m_BinState_left_point_list_keys);
            DisposeBuffer(ref m_BinState_left_point_list_values);
            DisposeBuffer(ref m_BinState_right_point_list_keys);
            DisposeBuffer(ref m_BinState_right_point_list_values);
           
            DisposeBuffer(ref m_ImageState_left_ranges);
            DisposeBuffer(ref m_ImageState_right_ranges);

            m_SplatCount = 0;
            m_GpuChunksValid = false;
        }

        public void OnDisable()
        {
            DisposeResourcesForAsset();
            NewGaussianSplatRenderSystem.instance.UnregisterSplat(this);
            m_Registered = false;
        }

        private void DebugMatrix(Matrix4x4 m, string name)
        {
            Debug.Log(string.Format("============================== {16} =================================\n{0}, {1}, {2}, {3}\n{4}, {5}, {6}, {7}\n{8}, {9}, {10}, {11}\n{12}, {13}, {14}, {15}\n==============================================================================",
                m.m00, m.m01, m.m02, m.m03,
                m.m10, m.m11, m.m12, m.m13,
                m.m20, m.m21, m.m22, m.m23,
                m.m30, m.m31, m.m32, m.m33, name));
        }

        internal void SetShaderKeywords(CommandBuffer cmb, bool isStereo)
        {
            if (m_IsSinglePass && isStereo)
            {
                cmb.EnableKeyword(m_CSSplatUtilities, m_SinglePassStereoKeyWord);
                cmb.DisableKeyword(m_CSSplatUtilities, m_SinglePassSingleKeyWord);
            }
            else if (m_IsSinglePass && !isStereo)
            {
                cmb.DisableKeyword(m_CSSplatUtilities, m_SinglePassStereoKeyWord);
                cmb.EnableKeyword(m_CSSplatUtilities, m_SinglePassSingleKeyWord);
            }
            else
            {
                cmb.DisableKeyword(m_CSSplatUtilities, m_SinglePassStereoKeyWord);
                cmb.DisableKeyword(m_CSSplatUtilities, m_SinglePassSingleKeyWord);
            }
        }

        Vector4 GetScreenParams(Camera cam)
        {
            int screenW = cam.pixelWidth, screenH = cam.pixelHeight;
            int eyeW = XRSettings.eyeTextureWidth, eyeH = XRSettings.eyeTextureHeight;
            return new Vector4((eyeW != 0 ? eyeW : screenW) * m_TextureScale, (eyeH != 0 ? eyeH : screenH) * m_TextureScale, m_WidthScale, m_HeightScale);
        }
        
        void GetTileConfig(ComputeShader cs, Camera cam, out int tileX, out int tileY, out int blockX, out int blockY)
        {
            if (!cs.IsSupported((int)KernelIndices.RenderViewData))
            {
                throw new Exception("Cannot get tile config, because render view data kernel is not supported.");
            }
            
            // 计算 tile 数量
            Vector4 screenPar = GetScreenParams(cam);
            cs.GetKernelThreadGroupSizes((int)KernelIndices.RenderViewData, out uint gsX, out uint gsY,
                out _);
            blockX = (int)gsX * m_TileScale;
            blockY = (int)gsY * m_TileScale;
            tileX = (int)((screenPar.x * m_WidthScale + blockX - 1) / blockX);
            tileY = (int)(screenPar.y * m_HeightScale + blockY - 1) / blockY;
        }

        void GetDispatchGroupNum(int count, int groupDim, out int countX, out int countY)
        {
            int limitX = groupDim * MAX_DISPATCH_GROUP;
            if (count > limitX)
            {
                countX = MAX_DISPATCH_GROUP;
                countY = count / limitX + 1;
            }
            else
            {
                countX = (count  + groupDim - 1) / groupDim;
                countY = 1;
            }
        }

        void UpdateOcclusionPlane()
        {
            // 更新 Occlusion Culling Plane
            for (int i = 0; i < m_OcclusionPlanes.Length; i++)
            {
                GameObject plane = m_OcclusionPlanes[i];
                Transform planeTransform = plane.transform;
                
                // 关闭 plane 的渲染
                // plane.GetComponent<MeshRenderer>().enabled = false;
                
                Vector3 planeNormal = planeTransform.up;
                Vector3 planeHeightDirection = planeTransform.forward;
                Vector3 planeWidthDirection = Vector3.Cross(planeNormal, planeHeightDirection);

                var planeData = new PlaneData()
                {
                    normal = planeNormal,
                    distance = Vector3.Dot(planeNormal, planeTransform.position),
                    center = planeTransform.position,
                    height = planeTransform.localScale.z * 10.0f,
                    width = planeTransform.localScale.x * 10.0f,
                    heightDir = planeHeightDirection,
                    widthDir = planeWidthDirection,
                };
                
                m_planeDataList[i] = planeData;
            }
            m_PlaneDataBuffer.SetData(m_planeDataList);
        }
        
        internal void PreProcessViewData(CommandBuffer cmb, Camera cam, TextureHandle depthTexture, Matrix4x4 preLeftViewProjectionMatrix, Matrix4x4 preRightViewProjectionMatrix, bool isStereo)
        {
            if (cam.cameraType == CameraType.Preview)
                return;

            var tr = transform;

            Matrix4x4 matView = cam.worldToCameraMatrix;
            Matrix4x4 matProj = GL.GetGPUProjectionMatrix(cam.projectionMatrix, true);
            Matrix4x4 matO2W = tr.localToWorldMatrix;
            Matrix4x4 matW2O = tr.worldToLocalMatrix;
            Vector4 screenPar = GetScreenParams(cam);
            Vector4 camPos = cam.transform.position;

            // calculate view dependent data for each splat
            SetAssetDataOnCS(cmb, KernelIndices.PreProcessViewData);
            
            // 初始化 NumArgs
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.InitNumArgs,Props.LeftNumArgs, m_NumArgs_left);
            if (m_IsSinglePass && isStereo)
            {
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.InitNumArgs,Props.RightNumArgs, m_NumArgs_right);
            }
            cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.InitNumArgs, 1, 1, 1);

            UpdateOcclusionPlane();
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.PreProcessViewData, Props.RO_PlaneData, m_PlaneDataBuffer);
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.NumOcclusionPlane, m_planeDataList.Length);
            
            // 设置 NumArgs
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.PreProcessViewData,Props.LeftNumArgs, m_NumArgs_left);
            if (m_IsSinglePass && isStereo)
            {
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.PreProcessViewData,Props.RightNumArgs, m_NumArgs_right);
            }

            // 设定 GemoState 的数据
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.PreProcessViewData, Props.GeomLeftData, m_GeomState_left_data);
            if (m_IsSinglePass && isStereo)
            {
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.PreProcessViewData, Props.GeomRightData, m_GeomState_right_data);
            }

            // 设定 tile 屏幕分块信息
            GetTileConfig(m_CSSplatUtilities, cam, out var tile_x, out var tile_y, out var block_x, out var block_y);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.TileConfig,
                new Vector4(block_x, block_y, tile_x, tile_y));
            
            // 设定深度剔除的阈值
            cmb.SetComputeFloatParam(m_CSSplatUtilities, Props.DepthCullingThreshold, m_DepthCullingThreshold);
            
            // 构造排序的 key【tile | depth】 和 value【coll_id】
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.PreProcessViewData,
                Props.BinLeftPointListKey, m_BinState_left_point_list_keys);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.PreProcessViewData,
                Props.BinLeftPointListValue, m_BinState_left_point_list_values);
            if (m_IsSinglePass && isStereo)
            {
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.PreProcessViewData,
                    Props.BinRightPointListKey, m_BinState_right_point_list_keys);
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.PreProcessViewData,
                    Props.BinRightPointListValue, m_BinState_right_point_list_values);
            }
            
            // 绑定上一帧的深度图
            cmb.SetComputeTextureParam(m_CSSplatUtilities, (int)KernelIndices.PreProcessViewData, Props.GSPreDepthTexture,
                depthTexture);
            
            // 绑定上一帧的 VP 矩阵
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixPreLeftVP, preLeftViewProjectionMatrix);
            if (m_IsSinglePass && isStereo)
            {
                cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixPreRightVP, preRightViewProjectionMatrix);
            }
            
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixMV, matView * matO2W);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixObjectToWorld, matO2W);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixWorldToObject, matW2O);

            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.VecScreenParams, screenPar);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.VecWorldSpaceCameraPos, camPos);
            cmb.SetComputeFloatParam(m_CSSplatUtilities, Props.SplatScale, m_SplatScale);
            cmb.SetComputeFloatParam(m_CSSplatUtilities, Props.SplatOpacityScale, m_OpacityScale);
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.SHOrder, m_SHOrder);
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.SHOnly, m_SHOnly ? 1 : 0);
            
            if (m_IsSinglePass)
            {
                Matrix4x4 matLView = cam.GetStereoViewMatrix(Camera.StereoscopicEye.Left);
                Matrix4x4 matRView = cam.GetStereoViewMatrix(Camera.StereoscopicEye.Right);
                cam.CopyStereoDeviceProjectionMatrixToNonJittered(Camera.StereoscopicEye.Left);
                Matrix4x4 matLProj =
                    GL.GetGPUProjectionMatrix(cam.GetStereoNonJitteredProjectionMatrix(Camera.StereoscopicEye.Left), true);
                cam.CopyStereoDeviceProjectionMatrixToNonJittered(Camera.StereoscopicEye.Right);
                Matrix4x4 matRProj =
                    GL.GetGPUProjectionMatrix(cam.GetStereoNonJitteredProjectionMatrix(Camera.StereoscopicEye.Right), true);
            
                cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixLV, matLView);
                cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixRV, matRView);
                cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixLP, matLProj);
                cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixRP, matRProj);
            }
            
            m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.PreProcessViewData, out uint gsX, out _,
                out _);
            
            GetDispatchGroupNum(m_SplatCount, (int)gsX, out int countX, out int countY);
            cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.PreProcessViewData,
                countX, countY, 1);
        }

        internal void RadixSortPoints(CommandBuffer cmb, Camera cam, bool isStereo)
        {
            if (cam.cameraType == CameraType.Preview)
                return;
            
            // 1. 初始化 Sort 参数 buffer
            cmb.SetComputeIntParam(m_CSSplatUtilities, "e_min", 2);
            cmb.SetComputeIntParam(m_CSSplatUtilities, "e_max", m_TileRenderCount);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.InitSortArgs, Props.LeftNumArgs, m_NumArgs_left);
            if (m_IsSinglePass && isStereo)
            {
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.InitSortArgs, Props.RightNumArgs, m_NumArgs_right);
            }
            cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.InitSortArgs, 1, 1, 1);
            
            // 2. 对 BinPointList 进行排序
            m_RadixSorter_left.Sort(
                cmb,
                m_NumArgs_left,
                m_BinState_left_point_list_keys,
                m_BinState_left_point_list_values,
                m_AltKey_left,
                m_AltPayload_left,
                m_GlobalHist_left,
                m_PassHist_left,
                true
                );

            if (m_IsSinglePass && isStereo)
            {
                m_RadixSorter_right.Sort(
                    cmb,
                    m_NumArgs_right,
                    m_BinState_right_point_list_keys,
                    m_BinState_right_point_list_values,
                    m_AltKey_right,
                    m_AltPayload_right,
                    m_GlobalHist_right,
                    m_PassHist_right,
                    true
                );
            }
            
            // 3. 计算 ImageState 数据
            {
                EnsureImageState(cam, isStereo);
                {
                    cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.IdentifyTileRanges, Props.RO_NumArgs, m_NumArgs_left);
                    cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.IdentifyTileRanges,
                        Props.RO_BinPointListKey, m_BinState_left_point_list_keys);
                    cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.IdentifyTileRanges,
                        Props.ImageRange, m_ImageState_left_ranges);
                    
                    cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.IdentifyTileRanges,
                        m_NumArgs_left, 20);

                    if (m_IsSinglePass && isStereo)
                    {
                        cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.IdentifyTileRanges, Props.RO_NumArgs, m_NumArgs_right);
                        cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.IdentifyTileRanges,
                            Props.RO_BinPointListKey, m_BinState_right_point_list_keys);
                        cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.IdentifyTileRanges,
                            Props.ImageRange, m_ImageState_right_ranges);
                    
                        cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.IdentifyTileRanges,
                            m_NumArgs_right, 20);
                    }
                }
            }
        }
        
        internal void RenderViewData(CommandBuffer cmb, Camera cam, TextureHandle gsRenderTexture, TextureHandle depthTexture, bool isStereo)
        {
            if (cam.cameraType == CameraType.Preview)
                return;
            
            // 设置 TileScale 分块放缩比例
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.TileScale, m_TileScale);
            
            // 设置屏幕像素大小
            Vector4 screenPar = GetScreenParams(cam);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.VecScreenParams, screenPar);

            // 设定 GemoState 的数据
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.RenderViewData, Props.RO_GeomLeftData,
                m_GeomState_left_data);
            if (m_IsSinglePass && isStereo)
            {
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.RenderViewData, Props.RO_GeomRightData,
                    m_GeomState_right_data);
            }

            // 设定 BinnState 的数据
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.RenderViewData,
                Props.RO_BinLeftPointListValue, m_BinState_left_point_list_values);
            if (m_IsSinglePass && isStereo)
            {
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.RenderViewData, Props.RO_BinRightPointListValue,
                    m_BinState_right_point_list_values);
            }


            // 设定 ImageState 的数据
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.RenderViewData, Props.RO_ImageLeftRange,
                m_ImageState_left_ranges);
            if (m_IsSinglePass && isStereo)
            {
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.RenderViewData, Props.RO_ImageRightRange,
                    m_ImageState_right_ranges);
            }

            // 设定输入 texture
            cmb.SetComputeTextureParam(m_CSSplatUtilities, (int)KernelIndices.RenderViewData, Props.GSRenderTexture,
                gsRenderTexture);
            cmb.SetComputeTextureParam(m_CSSplatUtilities, (int)KernelIndices.RenderViewData, Props.GSDepthTexture,
                depthTexture);
            
            if (m_IsSinglePass && !isStereo )
            {
                Matrix4x4 matLView = cam.GetStereoViewMatrix(Camera.StereoscopicEye.Left);
                Matrix4x4 matRView = cam.GetStereoViewMatrix(Camera.StereoscopicEye.Right);
                cam.CopyStereoDeviceProjectionMatrixToNonJittered(Camera.StereoscopicEye.Left);
                Matrix4x4 matLProj =
                    GL.GetGPUProjectionMatrix(cam.GetStereoNonJitteredProjectionMatrix(Camera.StereoscopicEye.Left), true);
                cam.CopyStereoDeviceProjectionMatrixToNonJittered(Camera.StereoscopicEye.Right);
                Matrix4x4 matRProj =
                    GL.GetGPUProjectionMatrix(cam.GetStereoNonJitteredProjectionMatrix(Camera.StereoscopicEye.Right), true);

                cmb.SetComputeFloatParam(m_CSSplatUtilities, Props.Focal, (float)(screenPar.y * 0.5 * matLProj.m11));
                cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixRPRVLV, matRProj * matRView * matLView.inverse);
            }
            
            GetTileConfig(m_CSSplatUtilities, cam, out var tile_x, out var tile_y, out _, out _);
            cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.RenderViewData,
                tile_x * m_TileScale, tile_y * m_TileScale, 1);
        }

        public void Update()
        {
            var curHash = m_Asset ? m_Asset.dataHash : new Hash128();
            if (m_PrevAsset != m_Asset || m_PrevHash != curHash)
            {
                m_PrevAsset = m_Asset;
                m_PrevHash = curHash;
                if (resourcesAreSetUp)
                {
                    DisposeResourcesForAsset();
                    CreateResourcesForAsset();
                }
                else
                {
                    Debug.LogError($"{nameof(GaussianSplatRenderer)} component is not set up correctly (Resource references are missing), or platform does not support compute shaders");
                }
            }
        }
    }
}