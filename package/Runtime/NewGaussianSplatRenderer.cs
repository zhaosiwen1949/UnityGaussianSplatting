// SPDX-License-Identifier: MIT

using System;
using GPUPrefixSums.Runtime;
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
    struct GemoData
    {
        public int4 touched_rects;
        public float4 conic_opacity;
        public float4 rgb;
        public float2 mean2D;
        public float depth;
        public float radius;
    };
    
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
    
    [ExecuteInEditMode]
    public class NewGaussianSplatRenderer : MonoBehaviour
    {
        public enum RenderMode
        {
            Splats,
            DebugPoints,
            DebugPointIndices,
            DebugBoxes,
            DebugChunkBounds,
        }
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

        public RenderMode m_RenderMode = RenderMode.Splats;
        [Range(1.0f,15.0f)] public float m_PointDisplaySize = 3.0f;

        public GaussianCutout[] m_Cutouts;

        public Shader m_ShaderNewRenderViewData;
        public Shader m_ShaderSplats;
        public Shader m_ShaderComposite;
        public Shader m_ShaderDebugPoints;
        public Shader m_ShaderDebugBoxes;
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
        internal GraphicsBuffer m_GpuView;
        internal GraphicsBuffer m_GpuIndexBuffer;
        internal GraphicsBuffer m_GpuFullScreenBuffer;
        
        // new tile-renderer needed buffer
        // VisibleCounts
        internal GraphicsBuffer m_VisibleCount;
        internal GraphicsBuffer m_VisibleBit;
        internal GraphicsBuffer m_VisibleBitOffset;
        
        // GeometryState
        internal GraphicsBuffer m_GeomState_splat_data;
        internal GraphicsBuffer m_GeomState_data;
        internal GraphicsBuffer m_GeomState_left_first_touched_tiles;
        internal GraphicsBuffer m_GeomState_right_first_touched_tiles;
        internal GraphicsBuffer m_GeomState_left_second_touched_tiles;
        internal GraphicsBuffer m_GeomState_right_second_touched_tiles;
        internal GraphicsBuffer m_GeomState_left_point_offsets;
        internal GraphicsBuffer m_GeomState_right_point_offsets;

        // BinningState
        internal GraphicsBuffer m_BinState_left_point_list_depth_keys;
        internal GraphicsBuffer m_BinState_left_point_list_tile_keys;
        internal GraphicsBuffer m_BinState_left_point_list_depth_values;
        internal GraphicsBuffer m_BinState_left_point_list_tile_values;
        internal GraphicsBuffer m_BinState_right_point_list_depth_keys;
        internal GraphicsBuffer m_BinState_right_point_list_tile_keys;
        internal GraphicsBuffer m_BinState_right_point_list_depth_values;
        internal GraphicsBuffer m_BinState_right_point_list_tile_values;

        // ImageState
        internal GraphicsBuffer m_ImageState_left_ranges;
        internal GraphicsBuffer m_ImageState_right_ranges;

        GpuSorting m_Sorter;
        GpuSorting.Args m_SorterArgs;
        private ReduceThenScan m_VisibleCountSumer;
        private ComputeBuffer m_VisibleCountReduction;
        private ReduceThenScan m_PrefixSumer;
        private ComputeBuffer m_ThreadBlockReduction;
        private GpuSorting m_FirstRadixSorter;
        private GpuSorting.Args m_FirstRadixSorterArgs;
        private GpuSorting m_SecondRadixSorter;
        private GpuSorting.Args m_SecondRadixSorterArgs;

        internal Material m_MatNewRenderViewData;
        internal Material m_MatSplats;
        internal Material m_MatComposite;
        internal Material m_MatDebugPoints;
        internal Material m_MatDebugBoxes;

        internal int m_FrameCounter;
        GaussianSplatAsset m_PrevAsset;
        Hash128 m_PrevHash;
        bool m_Registered;

        private int m_PreTileX, m_PreTileY;
        private int m_NumRendered;

        static readonly ProfilerMarker s_ProfSort = new(ProfilerCategory.Render, "GaussianSplat.Sort", MarkerFlags.SampleGPU);

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
            public static readonly int SplatViewData = Shader.PropertyToID("_SplatViewData");
            public static readonly int OrderBuffer = Shader.PropertyToID("_OrderBuffer");
            public static readonly int SplatScale = Shader.PropertyToID("_SplatScale");
            public static readonly int SplatOpacityScale = Shader.PropertyToID("_SplatOpacityScale");
            public static readonly int SplatSize = Shader.PropertyToID("_SplatSize");
            public static readonly int SplatCount = Shader.PropertyToID("_SplatCount");
            public static readonly int SHOrder = Shader.PropertyToID("_SHOrder");
            public static readonly int SHOnly = Shader.PropertyToID("_SHOnly");
            
            public static readonly int VisibleCount = Shader.PropertyToID("_VisibleCount");
            public static readonly int VisibleBit = Shader.PropertyToID("_VisibleBit");
            public static readonly int GeomSplatData = Shader.PropertyToID("_GeomSplatData");
            public static readonly int GeomData = Shader.PropertyToID("_GeomData");
            public static readonly int GeomLeftTouchedTiles = Shader.PropertyToID("_GeomLeftTouchedTiles");
            public static readonly int GeomRightTouchedTiles = Shader.PropertyToID("_GeomRightTouchedTiles");
            public static readonly int GeomFirstTouchedTiles = Shader.PropertyToID("_GeomFirstTouchedTiles");
            public static readonly int GeomSecondTouchedTiles = Shader.PropertyToID("_GeomSecondTouchedTiles");
            public static readonly int GeomPointOffset = Shader.PropertyToID("_GeomPointOffset");
            public static readonly int BinPointListDepthKey = Shader.PropertyToID("_BinPointListDepthKey");
            public static readonly int BinPointListTileKey = Shader.PropertyToID("_BinPointListTileKey");
            public static readonly int BinPointListDepthValue = Shader.PropertyToID("_BinPointListDepthValue");
            public static readonly int BinPointListTileValue = Shader.PropertyToID("_BinPointListTileValue");
            public static readonly int BinLeftPointListTileValue = Shader.PropertyToID("_BinLeftPointListTileValue");
            public static readonly int BinRightPointListTileValue = Shader.PropertyToID("_BinRightPointListTileValue");
            public static readonly int ImageRange = Shader.PropertyToID("_ImageRange");
            public static readonly int ImageLeftRange = Shader.PropertyToID("_ImageLeftRange");
            public static readonly int ImageRightRange = Shader.PropertyToID("_ImageRightRange");
            public static readonly int NumRendered = Shader.PropertyToID("_NumRendered");
            public static readonly int TileConfig = Shader.PropertyToID("_TileConfig");
            public static readonly int IterIndex = Shader.PropertyToID("_IterIndex");
            public static readonly int GSRenderTexture = Shader.PropertyToID("_GSRenderTexture");
            
            public static readonly int DisplayIndex = Shader.PropertyToID("_DisplayIndex");
            public static readonly int DisplayChunks = Shader.PropertyToID("_DisplayChunks");
            public static readonly int GaussianSplatRT = Shader.PropertyToID("_GaussianSplatRT");
            public static readonly int SplatSortKeys = Shader.PropertyToID("_SplatSortKeys");
            public static readonly int SplatSortDistances = Shader.PropertyToID("_SplatSortDistances");
            public static readonly int SrcBuffer = Shader.PropertyToID("_SrcBuffer");
            public static readonly int DstBuffer = Shader.PropertyToID("_DstBuffer");
            public static readonly int BufferSize = Shader.PropertyToID("_BufferSize");
            public static readonly int MatrixMV = Shader.PropertyToID("_MatrixMV");
            public static readonly int MatrixObjectToWorld = Shader.PropertyToID("_MatrixObjectToWorld");
            public static readonly int MatrixWorldToObject = Shader.PropertyToID("_MatrixWorldToObject");
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
        public int splatCount => m_SplatCount;

        enum KernelIndices
        {
            SetIndices,
            CalcDistances,
            CalcViewData,
            PreProcessViewData,
            DuplicateWithDepthKeys,
            ReorderTouchedTiles,
            DuplicateWithTileKeys,
            IdentifyTileRanges,
            InitImageRanges,
            CalcRanges,
            GetNumRendered,
            SetVisibleCounts,
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

        const int kGpuViewDataSize = 40;

        void CreateResourcesForAsset()
        {
            if (!HasValidAsset)
                return;

            m_SplatCount = asset.splatCount;
            m_TileRenderCount = 100 * asset.splatCount;
            m_GpuPosData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, (int) (asset.posData.dataSize / 4), 4) { name = "GaussianPosData" };
            m_GpuPosData.SetData(asset.posData.GetData<uint>());
            m_GpuOtherData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, (int) (asset.otherData.dataSize / 4), 4) { name = "GaussianOtherData" };
            m_GpuOtherData.SetData(asset.otherData.GetData<uint>());
            m_GpuSHData = new GraphicsBuffer(GraphicsBuffer.Target.Raw, (int) (asset.shData.dataSize / 4), 4) { name = "GaussianSHData" };
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

            m_GpuView = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_Asset.splatCount, kGpuViewDataSize) {name = "GaussianViewData"};
            m_GpuIndexBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Index, 36, 2);
            // cube indices, most often we use only the first quad
            m_GpuIndexBuffer.SetData(new ushort[]
            {
                0, 1, 2, 1, 3, 2,
                4, 6, 5, 5, 6, 7,
                0, 2, 4, 4, 2, 6,
                1, 5, 3, 5, 7, 3,
                0, 4, 1, 4, 5, 1,
                2, 3, 6, 3, 7, 6
            });
            m_GpuFullScreenBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Index, 3, 2);
            m_GpuFullScreenBuffer.SetData(new ushort[]
            {
                0, 1, 2
            });

            // InitRadixSortBuffers(splatCount);
            InitSortBuffers(splatCount);
            
            // 初始化 VisibleCounts
            m_VisibleCount = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1 ,4) { name = "GaussianSplatVisibleCount" };
            uint[] zero = { 0 };
            m_VisibleCount.SetData(zero);
            
            // 初始化 GeometryState
            int splatCountScale = 1;
            m_GeomState_data =
                new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_SplatCount * splatCountScale, 16 * 4)
                    { name = "GeomStateData" };
            
            m_GeomState_splat_data =
                new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_SplatCount * splatCountScale, 76 * 4)
                    { name = "GeomStateSplatData" };
            
            // 由于 PrefixSum 要求 Padding 数组数量到 4 的倍数，同时 stride 必须为 16 的倍数，所以我们把数组数量对齐到 GROUP_SIZE，同时保证 GROUP_ZISE 是 4 的倍数
            m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.PreProcessViewData, out uint gsX, out _,
                out _);
            int count = ((m_SplatCount + (int)gsX - 1) / (int)gsX) * (int)gsX / 4;
            m_GeomState_left_first_touched_tiles = new GraphicsBuffer(GraphicsBuffer.Target.Structured, count, 4 * 4)
                { name = "GeomStateLeftFirstTouchedTilesData" };
            m_GeomState_right_first_touched_tiles = new GraphicsBuffer(GraphicsBuffer.Target.Structured, count, 4 * 4)
                { name = "GeomStateRightFirstTouchedTilesData" };
            m_GeomState_left_second_touched_tiles = new GraphicsBuffer(GraphicsBuffer.Target.Structured, count, 4 * 4)
                { name = "GeomStateLeftSecondTouchedTilesData" };
            m_GeomState_right_second_touched_tiles = new GraphicsBuffer(GraphicsBuffer.Target.Structured, count, 4 * 4)
                { name = "GeomStateRightSecondTouchedTilesData" };
            m_GeomState_left_point_offsets = new GraphicsBuffer(GraphicsBuffer.Target.Structured, count, 4 * 4)
                { name = "GeomStateLeftPointOffsetData" };
            m_GeomState_right_point_offsets = new GraphicsBuffer(GraphicsBuffer.Target.Structured, count, 4 * 4)
                { name = "GeomStateRightPointOffsetData" };
            m_VisibleBit = new GraphicsBuffer(GraphicsBuffer.Target.Structured, count , 4 * 4) { name = "GaussianSplatVisibleBit" };
            m_VisibleBitOffset = new GraphicsBuffer(GraphicsBuffer.Target.Structured, count , 4 * 4) { name = "GaussianSplatVisibleBitOffset" };
            
            // BinState 中深度排序的部分，和点云点数保持相同
            m_BinState_left_point_list_depth_keys = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_SplatCount, 4)
                { name = "BinStateLeftDepthKeyData" };
            m_BinState_left_point_list_depth_values = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_SplatCount, 4)
                { name = "BinStateLeftDepthValueData" };
            m_BinState_left_point_list_tile_keys = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_TileRenderCount, 4)
                { name = "BinStateLeftTileKeyData" };
            m_BinState_left_point_list_tile_values = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_TileRenderCount, 4)
                { name = "BinStateLeftTileValueData" };
            
            InitRadixSortBuffers(splatCount);

        }

        public void EnsureImageState(Camera cam)
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
                    m_CSSplatUtilities.SetBuffer((int)KernelIndices.InitImageRanges, Props.ImageRange, m_ImageState_left_ranges);
                    m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.InitImageRanges, out uint gsX,
                        out _, out _);
                    int count = (tile_count + (int)gsX - 1) / (int)gsX;
                    m_CSSplatUtilities.Dispatch( (int)KernelIndices.InitImageRanges,
                        count, 1, 1);
                }
                
                m_PreTileX = tile_x;
                m_PreTileY = tile_y;
            }
        }
        
        void InitRadixSortBuffers(int splatCount)
        {
            // 初始化计算 VisibleCount 的前缀和算法
            DisposeBuffer(ref m_VisibleCountReduction);
            m_VisibleCountSumer = new ReduceThenScan(
                m_CSSplatUtilities,
                splatCount + 2048,
                ref m_VisibleCountReduction);
            
            // 初始化前缀和算法
            DisposeBuffer(ref m_ThreadBlockReduction);
            m_PrefixSumer = new ReduceThenScan(
                m_CSSplatUtilities,
                splatCount + 2048,
                ref m_ThreadBlockReduction);
            
            // 初始化第一遍的基数排序需要的资源
            m_FirstRadixSorterArgs.resources.Dispose();
            m_FirstRadixSorter = new GpuSorting(m_CSSplatUtilities);
            m_FirstRadixSorterArgs.inputKeys = m_BinState_left_point_list_depth_keys;
            m_FirstRadixSorterArgs.inputValues = m_BinState_left_point_list_depth_values;
            if (m_Sorter.Valid)
            {
                m_FirstRadixSorterArgs.resources = GpuSorting.SupportResources.Load((uint)splatCount);
            }
            
            // 初始化第二遍的基数排序需要的资源
            m_SecondRadixSorterArgs.resources.Dispose();
            m_SecondRadixSorter = new GpuSorting(m_CSSplatUtilities);
            m_SecondRadixSorterArgs.inputKeys = m_BinState_left_point_list_tile_keys;
            m_SecondRadixSorterArgs.inputValues = m_BinState_left_point_list_tile_values;
            if (m_Sorter.Valid)
            {
                m_SecondRadixSorterArgs.resources = GpuSorting.SupportResources.Load((uint)m_TileRenderCount );
            }
        }

        void InitSortBuffers(int count)
        {
            m_GpuSortDistances?.Dispose();
            m_GpuSortKeys?.Dispose();
            m_SorterArgs.resources.Dispose();

            EnsureSorterAndRegister();

            m_GpuSortDistances = new GraphicsBuffer(GraphicsBuffer.Target.Structured, count, 4) { name = "GaussianSplatSortDistances" };
            m_GpuSortKeys = new GraphicsBuffer(GraphicsBuffer.Target.Structured, count, 4) { name = "GaussianSplatSortIndices" };

            // init keys buffer to splat indices
            m_CSSplatUtilities.SetBuffer((int)KernelIndices.SetIndices, Props.SplatSortKeys, m_GpuSortKeys);
            m_CSSplatUtilities.SetInt(Props.SplatCount, m_GpuSortDistances.count);
            m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.SetIndices, out uint gsX, out _, out _);
            m_CSSplatUtilities.Dispatch((int)KernelIndices.SetIndices, (m_GpuSortDistances.count + (int)gsX - 1)/(int)gsX, 1, 1);

            m_SorterArgs.inputKeys = m_GpuSortDistances;
            m_SorterArgs.inputValues = m_GpuSortKeys;
            m_SorterArgs.count = (uint)count;
            if (m_Sorter.Valid)
                m_SorterArgs.resources = GpuSorting.SupportResources.Load((uint)count);
        }

        bool resourcesAreSetUp => m_ShaderNewRenderViewData != null && m_ShaderSplats != null && m_ShaderComposite != null && m_ShaderDebugPoints != null &&
                                  m_ShaderDebugBoxes != null && m_CSSplatUtilities != null && SystemInfo.supportsComputeShaders;

        public void EnsureMaterials()
        {
            if (m_MatSplats == null && resourcesAreSetUp)
            {
                m_MatNewRenderViewData = new Material(m_ShaderNewRenderViewData) { name = "GaussianNewRenderViewData" };
                m_MatSplats = new Material(m_ShaderSplats) {name = "GaussianSplats"};
                m_MatComposite = new Material(m_ShaderComposite) {name = "GaussianClearDstAlpha"};
                m_MatDebugPoints = new Material(m_ShaderDebugPoints) {name = "GaussianDebugPoints"};
                m_MatDebugBoxes = new Material(m_ShaderDebugBoxes) {name = "GaussianDebugBoxes"};
            }
        }

        public void EnsureSorterAndRegister()
        {
            if (m_Sorter == null && resourcesAreSetUp)
            {
                m_Sorter = new GpuSorting(m_CSSplatUtilities);
            }

            if (!m_Registered && resourcesAreSetUp)
            {
                NewGaussianSplatRenderSystem.instance.RegisterSplat(this);
                m_Registered = true;
            }
        }

        public void OnEnable()
        {
            m_FrameCounter = 0;
            if (!resourcesAreSetUp)
                return;

            EnsureMaterials();
            EnsureSorterAndRegister();

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
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatViewData, m_GpuView);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.OrderBuffer, m_GpuSortKeys);

            cmb.SetComputeIntParam(cs, Props.SplatBitsValid,  0);
            uint format = (uint)m_Asset.posFormat | ((uint)m_Asset.scaleFormat << 8) | ((uint)m_Asset.shFormat << 16);
            cmb.SetComputeIntParam(cs, Props.SplatFormat, (int)format);
            cmb.SetComputeIntParam(cs, Props.SplatCount, m_SplatCount);
            cmb.SetComputeIntParam(cs, Props.SplatChunkCount, m_GpuChunksValid ? m_GpuChunks.count : 0);

            cmb.SetComputeIntParam(cs, Props.SplatCutoutsCount, m_Cutouts?.Length ?? 0);
        }

        internal void SetAssetDataOnMaterial(MaterialPropertyBlock mat)
        {
            mat.SetBuffer(Props.SplatPos, m_GpuPosData);
            mat.SetBuffer(Props.SplatOther, m_GpuOtherData);
            mat.SetBuffer(Props.SplatSH, m_GpuSHData);
            mat.SetTexture(Props.SplatColor, m_GpuColorData);
            mat.SetBuffer(Props.SplatSelectedBits, m_GpuPosData);
            mat.SetBuffer(Props.SplatDeletedBits, m_GpuPosData);
            mat.SetInt(Props.SplatBitsValid, 0);
            uint format = (uint)m_Asset.posFormat | ((uint)m_Asset.scaleFormat << 8) | ((uint)m_Asset.shFormat << 16);
            mat.SetInteger(Props.SplatFormat, (int)format);
            mat.SetInteger(Props.SplatCount, m_SplatCount);
            mat.SetInteger(Props.SplatChunkCount, m_GpuChunksValid ? m_GpuChunks.count : 0);
        }

        static void DisposeBuffer(ref GraphicsBuffer buf)
        {
            buf?.Dispose();
            buf = null;
        }
        
        static void DisposeBuffer(ref ComputeBuffer buf)
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

            DisposeBuffer(ref m_GpuView);
            DisposeBuffer(ref m_GpuIndexBuffer);
            DisposeBuffer(ref m_GpuFullScreenBuffer);
            DisposeBuffer(ref m_GpuSortDistances);
            DisposeBuffer(ref m_GpuSortKeys);

            m_SorterArgs.resources.Dispose();
            DisposeBuffer(ref m_ThreadBlockReduction);
            DisposeBuffer(ref m_VisibleCountReduction);
            m_FirstRadixSorterArgs.resources.Dispose();
            m_SecondRadixSorterArgs.resources.Dispose();

            DisposeBuffer(ref m_VisibleCount);
            DisposeBuffer(ref m_VisibleBit);
            DisposeBuffer(ref m_VisibleBitOffset);
            DisposeBuffer(ref m_GeomState_data);
            DisposeBuffer(ref m_GeomState_splat_data);
            DisposeBuffer(ref m_GeomState_left_first_touched_tiles);
            DisposeBuffer(ref m_GeomState_right_first_touched_tiles);
            DisposeBuffer(ref m_GeomState_left_second_touched_tiles);
            DisposeBuffer(ref m_GeomState_right_second_touched_tiles);
            DisposeBuffer(ref m_GeomState_left_point_offsets);
            DisposeBuffer(ref m_GeomState_right_point_offsets);

            DisposeBuffer(ref m_BinState_left_point_list_depth_keys);
            DisposeBuffer(ref m_BinState_left_point_list_tile_keys);
            DisposeBuffer(ref m_BinState_left_point_list_depth_values);
            DisposeBuffer(ref m_BinState_left_point_list_tile_values);
            DisposeBuffer(ref m_BinState_right_point_list_depth_keys);
            DisposeBuffer(ref m_BinState_right_point_list_tile_keys);
            DisposeBuffer(ref m_BinState_right_point_list_depth_values);
            DisposeBuffer(ref m_BinState_right_point_list_tile_values);
           
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

            DestroyImmediate(m_MatSplats);
            DestroyImmediate(m_MatComposite);
            DestroyImmediate(m_MatDebugPoints);
            DestroyImmediate(m_MatDebugBoxes);
        }
        
        void GetTileConfig(ComputeShader cs, Camera cam, out int tileX, out int tileY, out int blockX, out int blockY)
        {
            if (!cs.IsSupported((int)KernelIndices.RenderViewData))
            {
                throw new Exception("Cannot get tile config, because render view data kernel is not supported.");
            }
            
            // 计算 tile 数量
            int screenW = cam.pixelWidth, screenH = cam.pixelHeight;
            int eyeW = XRSettings.eyeTextureWidth, eyeH = XRSettings.eyeTextureHeight;
            int screen_width = eyeW != 0 ? eyeW : screenW;
            int screen_height = eyeH != 0 ? eyeH : screenH;
            cs.GetKernelThreadGroupSizes((int)KernelIndices.RenderViewData, out uint gsX, out uint gsY,
                out _);
            blockX = (int)gsX;
            blockY = (int)gsY;
            tileX = (screen_width + blockX - 1) / blockX;
            tileY = (screen_height + blockY - 1) / blockY;
        }
        
        internal void PreProcessViewData(CommandBuffer cmb, Camera cam)
        {
            if (cam.cameraType == CameraType.Preview)
                return;

            var tr = transform;

            Matrix4x4 matView = cam.worldToCameraMatrix;
            Matrix4x4 matO2W = tr.localToWorldMatrix;
            Matrix4x4 matW2O = tr.worldToLocalMatrix;
            int screenW = cam.pixelWidth, screenH = cam.pixelHeight;
            int eyeW = XRSettings.eyeTextureWidth, eyeH = XRSettings.eyeTextureHeight;
            Vector4 screenPar = new Vector4(eyeW != 0 ? eyeW : screenW, eyeH != 0 ? eyeH : screenH, 0, 0);
            Vector4 camPos = cam.transform.position;

            // calculate view dependent data for each splat
            SetAssetDataOnCS(cmb, KernelIndices.PreProcessViewData);
            
            // 初始化 VisibleCount
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.SetVisibleCounts,Props.VisibleCount, m_VisibleCount);
            cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.SetVisibleCounts,
                1, 1, 1);
            // 设置 VisibleCount 和 VisibleBit
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.PreProcessViewData,Props.VisibleCount, m_VisibleCount);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.PreProcessViewData,Props.VisibleBit, m_VisibleBit);
            
            // 设定 GemoState 的数据
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.PreProcessViewData, Props.GeomSplatData, m_GeomState_splat_data);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.PreProcessViewData, Props.GeomData, m_GeomState_data);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.PreProcessViewData,  Props.GeomLeftTouchedTiles, m_GeomState_left_first_touched_tiles);
            // cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.PreProcessViewData,
            //     Props.GeomRightTouchedTiles, m_GeomState_right_touched_tiles);

            //设定 tile 屏幕分块信息
            GetTileConfig(m_CSSplatUtilities, cam, out var tile_x, out var tile_y, out var block_x, out var block_y);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.TileConfig,
                new Vector4(block_x, block_y, tile_x, tile_y));

            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixMV, matView * matO2W);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixObjectToWorld, matO2W);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixWorldToObject, matW2O);

            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.VecScreenParams, screenPar);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.VecWorldSpaceCameraPos, camPos);
            cmb.SetComputeFloatParam(m_CSSplatUtilities, Props.SplatScale, m_SplatScale);
            cmb.SetComputeFloatParam(m_CSSplatUtilities, Props.SplatOpacityScale, m_OpacityScale);
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.SHOrder, m_SHOrder);
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.SHOnly, m_SHOnly ? 1 : 0);
            // if (XRSettings.stereoRenderingMode == XRSettings.StereoRenderingMode.SinglePassInstanced)
            // if (IsSinglePass)
            // {
            //     Matrix4x4 matLView = cam.GetStereoViewMatrix(Camera.StereoscopicEye.Left);
            //     Matrix4x4 matRView = cam.GetStereoViewMatrix(Camera.StereoscopicEye.Right);
            //     Matrix4x4 matLProj =
            //         GL.GetGPUProjectionMatrix(cam.GetStereoProjectionMatrix(Camera.StereoscopicEye.Left), false);
            //     Matrix4x4 matRProj =
            //         GL.GetGPUProjectionMatrix(cam.GetStereoProjectionMatrix(Camera.StereoscopicEye.Right), false);
            //     Vector4 cameraLPos = matLView.GetPosition();
            //     Vector4 cameraRPos = matRView.GetPosition();
            //
            //     cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixLV, matLView);
            //     cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixRV, matRView);
            //     cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixLP, matLProj);
            //     cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixRP, matRProj);
            //     cmb.SetComputeIntParam(m_CSSplatUtilities, Props.SinglePassMode, 1);
            // }
            // else
            // {
            //     cmb.SetComputeIntParam(m_CSSplatUtilities, Props.SinglePassMode, 0);
            // }

            // int splatCountScale = Application.isPlaying ? 2 : 1;
            // int splatCountScale = IsSinglePass ? 2 : 1;
            int splatCountScale = 1;
            m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.PreProcessViewData, out uint gsX, out _,
                out _);
            int count = (m_SplatCount * splatCountScale + (int)gsX - 1) / (int)gsX;
            cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.PreProcessViewData,
                count, 1, 1);
        }

        internal void RadixSortPoints(CommandBuffer cmb, Camera cam)
        {
            if (cam.cameraType == CameraType.Preview)
                return;
            
            // 1. 销毁分配的 GraphicsBuffer
            // DisposeBuffer(ref m_BinState_left_point_list_tile_keys);
            // DisposeBuffer(ref m_BinState_left_point_list_tile_values);

            int visibleCount = 2;
            // 1. 获取 VisibleCount
            {
                m_VisibleCountSumer.PrefixSumInclusive(
                    cmb,
                    m_SplatCount,
                    m_VisibleBit,
                    m_VisibleBitOffset,
                    m_VisibleCountReduction);
                
                var visiblebit_offset_list = new Int4Data[1];
                int splat_count = m_SplatCount - 1;
                int list_index = splat_count / 4;
                int data_index = splat_count % 4;
                m_VisibleBitOffset.GetData(visiblebit_offset_list, 0, list_index, 1);
                visibleCount = Math.Min(Math.Max(visiblebit_offset_list[0].GetElement(data_index), visibleCount), 65535 * 1024);
                Debug.Log($"bitCounts: {visibleCount}");
                
                // uint[] visibleCountList = new uint[1];
                // m_VisibleCount.GetData(visibleCountList);
                // Debug.Log($"visibleCounts: {visibleCountList[0]}");
                Debug.Log("m_SplatCount: " + m_SplatCount);
            }
            visibleCount = m_SplatCount;
            
            // 2. 构造第一次排序的 key【depth】 和 value【coll_id】
            {
                cmb.SetComputeIntParam(m_CSSplatUtilities, Props.SplatCount, visibleCount);

                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.DuplicateWithDepthKeys, Props.GeomData,
                    m_GeomState_data);

                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.DuplicateWithDepthKeys,
                    Props.BinPointListDepthKey, m_BinState_left_point_list_depth_keys);
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.DuplicateWithDepthKeys,
                    Props.BinPointListDepthValue, m_BinState_left_point_list_depth_values);

                m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.DuplicateWithDepthKeys,
                    out uint gsX, out _, out _);
                int count = (visibleCount + (int)gsX - 1) / (int)gsX;
                cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.DuplicateWithDepthKeys,
                    count, 1, 1);
            }
            
            // 3. 第一次基于深度进行排序
            // m_FirstRadixSorterArgs.inputKeys = m_BinState_left_point_list_depth_keys;
            // m_FirstRadixSorterArgs.inputValues = m_BinState_left_point_list_depth_values;
            // m_FirstRadixSorterArgs.count = (uint)m_SplatCount;
            // if (m_Sorter.Valid)
            // {
            //     // 初始化第一遍的基数排序
            //     m_FirstRadixSorterArgs.resources.Dispose();
            //     m_FirstRadixSorterArgs.resources = GpuSorting.SupportResources.Load(m_FirstRadixSorterArgs.count);
            //     m_FirstRadixSorter.Dispatch(cmb, m_FirstRadixSorterArgs);
            // }
            {
                m_FirstRadixSorterArgs.count = (uint)visibleCount;
                m_FirstRadixSorter.Dispatch(cmb, m_FirstRadixSorterArgs);
            }
            
            // 4. 根据排序后的 id，重新组织 touched_tiles 数组的顺序
            {
                cmb.SetComputeIntParam(m_CSSplatUtilities, Props.SplatCount, visibleCount);

                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.ReorderTouchedTiles,
                    Props.GeomFirstTouchedTiles, m_GeomState_left_first_touched_tiles);
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.ReorderTouchedTiles,
                    Props.BinPointListDepthValue, m_BinState_left_point_list_depth_values);
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.ReorderTouchedTiles,
                    Props.GeomSecondTouchedTiles, m_GeomState_left_second_touched_tiles);

                m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.ReorderTouchedTiles,
                    out uint gsX, out _, out _);
                int count = (visibleCount + (int)gsX - 1) / (int)gsX;
                cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.ReorderTouchedTiles,
                    count, 1, 1);
            }
            
            // 5. 求前缀和数组
            {
                m_PrefixSumer.PrefixSumInclusive(
                    cmb,
                    visibleCount,
                    m_GeomState_left_second_touched_tiles,
                    m_GeomState_left_point_offsets,
                    m_ThreadBlockReduction);
            }
            
            // 6. 分配 BinnState 数据
            {
                var number_rendered_list = new Int4Data[1];
                int splat_count = visibleCount - 1;
                int list_index = splat_count / 4;
                int data_index = splat_count % 4;
                m_GeomState_left_point_offsets.GetData(number_rendered_list, 0, list_index, 1);
                m_NumRendered = number_rendered_list[0].GetElement(data_index);

                if (m_NumRendered <= 0 || m_NumRendered >= 536870912)
                {
                    m_NumRendered = 1;
                }

                m_NumRendered = Math.Min(m_NumRendered, m_TileRenderCount);
                Debug.Log("m_NumRendered: " + m_NumRendered);
            
                // m_BinState_left_point_list_tile_keys = new GraphicsBuffer(GraphicsBuffer.Target.Structured, number_rendered, 4)
                //     { name = "BinStateLeftTileKeyData" };
                // m_BinState_left_point_list_tile_values = new GraphicsBuffer(GraphicsBuffer.Target.Structured, number_rendered, 4)
                //     { name = "BinStateLeftTileValueData" };
            }
            
            // 7. 构造第二次排序的 key【tilekey】 和 value【coll_id】
            {
                cmb.SetComputeIntParam(m_CSSplatUtilities, Props.SplatCount, visibleCount);
            
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.DuplicateWithTileKeys, Props.GeomData,
                    m_GeomState_data);
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.DuplicateWithTileKeys,
                    Props.BinPointListDepthValue, m_BinState_left_point_list_depth_values);
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.DuplicateWithTileKeys,
                    Props.GeomPointOffset, m_GeomState_left_point_offsets);
                
                GetTileConfig(m_CSSplatUtilities, cam, out var tile_x, out var tile_y, out var block_x, out var block_y);
                cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.TileConfig,
                    new Vector4(block_x, block_y, tile_x, tile_y));
            
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.DuplicateWithTileKeys,
                    Props.BinPointListTileKey, m_BinState_left_point_list_tile_keys);
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.DuplicateWithTileKeys,
                    Props.BinPointListTileValue, m_BinState_left_point_list_tile_values);
            
                m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.DuplicateWithTileKeys,
                    out uint gsX, out _, out _);
                int count = (visibleCount + (int)gsX - 1) / (int)gsX;
                cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.DuplicateWithTileKeys,
                    count, 1, 1);
            }
            
            // 8. 第二次基数排序
            // m_SecondRadixSorterArgs.inputKeys = m_BinState_left_point_list_tile_keys;
            // m_SecondRadixSorterArgs.inputValues = m_BinState_left_point_list_tile_values;
            // // m_SecondRadixSorterArgs.count = (uint)number_rendered;
            // m_SecondRadixSorterArgs.count = (uint)m_NumRendered;
            // if (m_Sorter.Valid)
            // {
            //     // 初始化第一遍的基数排序
            //     m_SecondRadixSorterArgs.resources.Dispose();
            //     m_SecondRadixSorterArgs.resources = GpuSorting.SupportResources.Load(m_SecondRadixSorterArgs.count);
            //     m_SecondRadixSorter.Dispatch(cmb, m_SecondRadixSorterArgs);
            // }
            {
                m_SecondRadixSorterArgs.count = (uint)m_NumRendered;
                m_SecondRadixSorter.Dispatch(cmb, m_SecondRadixSorterArgs);
            }
            
            // 9. 计算 ImageState 数据
            {
                EnsureImageState(cam);
                {
                    // Debug.Log("number_rendered: " + number_rendered);
                    // cmb.SetComputeIntParam(m_CSSplatUtilities, Props.NumRendered, number_rendered);
                    cmb.SetComputeIntParam(m_CSSplatUtilities, Props.NumRendered, m_NumRendered);
                    cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.IdentifyTileRanges,
                        Props.BinPointListTileKey, m_BinState_left_point_list_tile_keys);
                    cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.IdentifyTileRanges,
                        Props.ImageRange, m_ImageState_left_ranges);
                
                    m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.IdentifyTileRanges, out uint gsX,
                        out _, out _);
                    // int count = (number_rendered + (int)gsX - 1) / (int)gsX;
                    int count = (m_NumRendered + (int)gsX - 1) / (int)gsX;
                    cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.IdentifyTileRanges,
                        count, 1, 1);
                }
                
                // // DEBUG: 计算 ImageState 中 range 的和
                // {
                //     GetTileConfig(m_CSSplatUtilities, cam, out var tile_x, out var tile_y, out _, out _);
                //     int tile_count = tile_x * tile_y;
                //     // Debug.Log("tile_x: " + tile_x + ", tile_y: " + tile_y);
                //     cmb.SetComputeIntParam(m_CSSplatUtilities, Props.NumRendered, tile_count);
                //     cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcRanges,
                //         Props.ImageLeftRange, m_ImageState_left_ranges);
                //     cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcRanges,
                //         Props.ImageRightRange, m_ImageState_right_ranges);
                //     
                //     m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.CalcRanges, out uint gsX,
                //         out _, out _);
                //     int count = (tile_count + (int)gsX - 1) / (int)gsX;
                //     cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.CalcRanges,
                //         count, 1, 1);
                //     
                //     Int2Data[] ImageRangeData = new Int2Data[m_ImageState_right_ranges.count];
                //     m_ImageState_right_ranges.GetData(ImageRangeData);
                //     int maxValue = 0;
                //     int maxIndex = 0;
                //     for (int i = 0; i < ImageRangeData.Length; ++i)
                //     {
                //         int currentValue = (int)ImageRangeData[i].x;
                //         int currentIndex = (int)ImageRangeData[i].y;
                //         if (currentValue > maxValue)
                //         {
                //             maxValue = currentValue;
                //             maxIndex = currentIndex;
                //         }
                //     }
                //     // Debug.Log("TileCount: " + tile_count + "; MaxValue: " + maxValue + "; MaxIndex: " + maxIndex);
                // }
            }
        }
        
        internal void RenderViewData(CommandBuffer cmb, Camera cam, TextureHandle gsRenderTexture)
        {
            if (cam.cameraType == CameraType.Preview)
                return;

            int screenW = cam.pixelWidth, screenH = cam.pixelHeight;
            int eyeW = XRSettings.eyeTextureWidth, eyeH = XRSettings.eyeTextureHeight;
            Vector4 screenPar = new Vector4(eyeW != 0 ? eyeW : screenW, eyeH != 0 ? eyeH : screenH, 0, 0);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.VecScreenParams, screenPar);

            // 设定 GemoState 的数据
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.RenderViewData, Props.GeomData,
                m_GeomState_data);

            // 设定 BinnState 的数据
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.RenderViewData,
                Props.BinLeftPointListTileValue, m_BinState_left_point_list_tile_values);
            // cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.RenderViewData,
            //     Props.BinRightPointListTileValue, m_BinState_right_point_list_tile_values);
            // cmb.SetComputeIntParam(m_CSSplatUtilities, Props.NumRendered, m_BinState_left_point_list_tile_values.count);
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.NumRendered, m_NumRendered);


            // 设定 ImageState 的数据
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.RenderViewData, Props.ImageLeftRange,
                m_ImageState_left_ranges);
            // cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.RenderViewData, Props.ImageRightRange,
            //     m_ImageState_right_ranges);

            // 设定输入 texture
            cmb.SetComputeTextureParam(m_CSSplatUtilities, (int)KernelIndices.RenderViewData, Props.GSRenderTexture,
                gsRenderTexture);
            
            m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.RenderViewData, out uint gsX, out uint gsY,
                out _);
            int count_x = ((int)screenPar.x + (int)gsX - 1) / (int)gsX;

            int count_y = ((int)screenPar.y + (int)gsY - 1) / (int)gsY;

            cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.RenderViewData,
                count_x, count_y, 1);
        }
        
        // internal void NewRenderViewData(CommandBuffer cmb, Camera cam, MaterialPropertyBlock mpb, TextureHandle gsRenderTexture)
        internal void NewRenderViewData(CommandBuffer cmb, Camera cam, MaterialPropertyBlock mpb)
        {
            if (cam.cameraType == CameraType.Preview)
                return;

            int screenW = cam.pixelWidth, screenH = cam.pixelHeight;
            int eyeW = XRSettings.eyeTextureWidth, eyeH = XRSettings.eyeTextureHeight;
            Vector4 screenPar = new Vector4(eyeW != 0 ? eyeW : screenW, eyeH != 0 ? eyeH : screenH, 0, 0);
            mpb.SetVector( Props.VecScreenParams, screenPar);

            // 设定 GemoState 的数据
            mpb.SetBuffer(Props.GeomData,
                m_GeomState_data);

            // 设定 BinnState 的数据
            mpb.SetBuffer(Props.BinLeftPointListTileValue, m_BinState_left_point_list_tile_values);
            // mpb.SetBuffer(Props.BinRightPointListTileValue, m_BinState_right_point_list_tile_values);
            mpb.SetInt(Props.NumRendered, m_NumRendered);


            // 设定 ImageState 的数据
            mpb.SetBuffer(Props.ImageLeftRange, m_ImageState_left_ranges);
            // mpb.SetBuffer(Props.ImageRightRange, m_ImageState_right_ranges);

            // mpb.SetTexture(Props.GSRenderTexture, gsRenderTexture);
            
            int indexCount = 3;
            int instanceCount = 1;
            MeshTopology topology = MeshTopology.Triangles;
            
            cmb.DrawProcedural(m_GpuFullScreenBuffer, Matrix4x4.identity, m_MatNewRenderViewData, 0, topology, indexCount, instanceCount, mpb);
        }
        
        internal void CalcViewData(CommandBuffer cmb, Camera cam)
        {
            if (cam.cameraType == CameraType.Preview)
                return;

            var tr = transform;

            Matrix4x4 matView = cam.worldToCameraMatrix;
            Matrix4x4 matO2W = tr.localToWorldMatrix;
            Matrix4x4 matW2O = tr.worldToLocalMatrix;
            int screenW = cam.pixelWidth, screenH = cam.pixelHeight;
            int eyeW = XRSettings.eyeTextureWidth, eyeH = XRSettings.eyeTextureHeight;
            Vector4 screenPar = new Vector4(eyeW != 0 ? eyeW : screenW, eyeH != 0 ? eyeH : screenH, 0, 0);
            Vector4 camPos = cam.transform.position;

            // calculate view dependent data for each splat
            SetAssetDataOnCS(cmb, KernelIndices.CalcViewData);

            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixMV, matView * matO2W);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixObjectToWorld, matO2W);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixWorldToObject, matW2O);

            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.VecScreenParams, screenPar);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.VecWorldSpaceCameraPos, camPos);
            cmb.SetComputeFloatParam(m_CSSplatUtilities, Props.SplatScale, m_SplatScale);
            cmb.SetComputeFloatParam(m_CSSplatUtilities, Props.SplatOpacityScale, m_OpacityScale);
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.SHOrder, m_SHOrder);
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.SHOnly, m_SHOnly ? 1 : 0);
            
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcViewData, Props.GeomData, m_GeomState_data);
            // cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcViewData,
            //     Props.GeomRightTouchedTiles, m_GeomState_right_touched_tiles);

            m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.CalcViewData, out uint gsX, out _, out _);
            cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.CalcViewData, (m_GpuView.count + (int)gsX - 1)/(int)gsX, 1, 1);
        }

        internal void SortPoints(CommandBuffer cmd, Camera cam, Matrix4x4 matrix)
        {
            if (cam.cameraType == CameraType.Preview)
                return;

            Matrix4x4 worldToCamMatrix = cam.worldToCameraMatrix;
            worldToCamMatrix.m20 *= -1;
            worldToCamMatrix.m21 *= -1;
            worldToCamMatrix.m22 *= -1;

            // calculate distance to the camera for each splat
            cmd.BeginSample(s_ProfSort);
            cmd.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcDistances, Props.SplatSortDistances, m_GpuSortDistances);
            cmd.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcDistances, Props.SplatSortKeys, m_GpuSortKeys);
            cmd.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcDistances, Props.SplatChunks, m_GpuChunks);
            cmd.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcDistances, Props.SplatPos, m_GpuPosData);
            cmd.SetComputeIntParam(m_CSSplatUtilities, Props.SplatFormat, (int)m_Asset.posFormat);
            cmd.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixMV, worldToCamMatrix * matrix);
            cmd.SetComputeIntParam(m_CSSplatUtilities, Props.SplatCount, m_SplatCount);
            cmd.SetComputeIntParam(m_CSSplatUtilities, Props.SplatChunkCount, m_GpuChunksValid ? m_GpuChunks.count : 0);
            m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.CalcDistances, out uint gsX, out _, out _);
            cmd.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.CalcDistances, (m_GpuSortDistances.count + (int)gsX - 1)/(int)gsX, 1, 1);

            // sort the splats
            EnsureSorterAndRegister();
            m_Sorter.Dispatch(cmd, m_SorterArgs);
            cmd.EndSample(s_ProfSort);
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

            if (Input.GetKeyDown(KeyCode.Q))
            {
                var cam = m_Asset.cameras[0];
                var selfTr = transform;
                var camTr = Camera.main.transform;
                var prevParent = camTr.parent;
                Camera.main.transform.parent = selfTr;
                Camera.main.transform.localPosition = cam.pos;
                Camera.main.transform.localRotation = Quaternion.LookRotation(cam.axisZ, cam.axisY);
                Camera.main.transform.parent = prevParent;
            }
            
            if (Input.GetKeyDown(KeyCode.W))
            {
                var cam = m_Asset.cameras[1];
                var selfTr = transform;
                var camTr = Camera.main.transform;
                var prevParent = camTr.parent;
                Camera.main.transform.parent = selfTr;
                Camera.main.transform.localPosition = cam.pos;
                Camera.main.transform.localRotation = Quaternion.LookRotation(cam.axisZ, cam.axisY);
                Camera.main.transform.parent = prevParent;
            }
            
            if (Input.GetKeyDown(KeyCode.E))
            {
                var cam = m_Asset.cameras[2];
                var selfTr = transform;
                var camTr = Camera.main.transform;
                var prevParent = camTr.parent;
                Camera.main.transform.parent = selfTr;
                Camera.main.transform.localPosition = cam.pos;
                Camera.main.transform.localRotation = Quaternion.LookRotation(cam.axisZ, cam.axisY);
                Camera.main.transform.parent = prevParent;
            }
            
            if (Input.GetKeyDown(KeyCode.R))
            {
                var cam = m_Asset.cameras[3];
                var selfTr = transform;
                var camTr = Camera.main.transform;
                var prevParent = camTr.parent;
                Camera.main.transform.parent = selfTr;
                Camera.main.transform.localPosition = cam.pos;
                Camera.main.transform.localRotation = Quaternion.LookRotation(cam.axisZ, cam.axisY);
                Camera.main.transform.parent = prevParent;
            }
            
            if (Input.GetKeyDown(KeyCode.A))
            {
                var cam = m_Asset.cameras[4];
                var selfTr = transform;
                var camTr = Camera.main.transform;
                var prevParent = camTr.parent;
                Camera.main.transform.parent = selfTr;
                Camera.main.transform.localPosition = cam.pos;
                Camera.main.transform.localRotation = Quaternion.LookRotation(cam.axisZ, cam.axisY);
                Camera.main.transform.parent = prevParent;
            }
            
            if (Input.GetKeyDown(KeyCode.S))
            {
                var cam = m_Asset.cameras[5];
                var selfTr = transform;
                var camTr = Camera.main.transform;
                var prevParent = camTr.parent;
                Camera.main.transform.parent = selfTr;
                Camera.main.transform.localPosition = cam.pos;
                Camera.main.transform.localRotation = Quaternion.LookRotation(cam.axisZ, cam.axisY);
                Camera.main.transform.parent = prevParent;
            }
            
            if (Input.GetKeyDown(KeyCode.D))
            {
                var cam = m_Asset.cameras[6];
                var selfTr = transform;
                var camTr = Camera.main.transform;
                var prevParent = camTr.parent;
                Camera.main.transform.parent = selfTr;
                Camera.main.transform.localPosition = cam.pos;
                Camera.main.transform.localRotation = Quaternion.LookRotation(cam.axisZ, cam.axisY);
                Camera.main.transform.parent = prevParent;
            }
            
            if (Input.GetKeyDown(KeyCode.F))
            {
                var cam = m_Asset.cameras[7];
                var selfTr = transform;
                var camTr = Camera.main.transform;
                var prevParent = camTr.parent;
                Camera.main.transform.parent = selfTr;
                Camera.main.transform.localPosition = cam.pos;
                Camera.main.transform.localRotation = Quaternion.LookRotation(cam.axisZ, cam.axisY);
                Camera.main.transform.parent = prevParent;
            }
            
            if (Input.GetKeyDown(KeyCode.Z))
            {
                var cam = m_Asset.cameras[8];
                var selfTr = transform;
                var camTr = Camera.main.transform;
                var prevParent = camTr.parent;
                Camera.main.transform.parent = selfTr;
                Camera.main.transform.localPosition = cam.pos;
                Camera.main.transform.localRotation = Quaternion.LookRotation(cam.axisZ, cam.axisY);
                Camera.main.transform.parent = prevParent;
            }

            if (Input.GetKeyDown(KeyCode.X))
            {
                var cam = m_Asset.cameras[9];
                var selfTr = transform;
                var camTr = Camera.main.transform;
                var prevParent = camTr.parent;
                Camera.main.transform.parent = selfTr;
                Camera.main.transform.localPosition = cam.pos;
                Camera.main.transform.localRotation = Quaternion.LookRotation(cam.axisZ, cam.axisY);
                Camera.main.transform.parent = prevParent;
            }
            
            if (Input.GetKeyDown(KeyCode.C))
            {
                var cam = m_Asset.cameras[10];
                var selfTr = transform;
                var camTr = Camera.main.transform;
                var prevParent = camTr.parent;
                Camera.main.transform.parent = selfTr;
                Camera.main.transform.localPosition = cam.pos;
                Camera.main.transform.localRotation = Quaternion.LookRotation(cam.axisZ, cam.axisY);
                Camera.main.transform.parent = prevParent;
            }
            
            if (Input.GetKeyDown(KeyCode.V))
            {
                var cam = m_Asset.cameras[11];
                var selfTr = transform;
                var camTr = Camera.main.transform;
                var prevParent = camTr.parent;
                Camera.main.transform.parent = selfTr;
                Camera.main.transform.localPosition = cam.pos;
                Camera.main.transform.localRotation = Quaternion.LookRotation(cam.axisZ, cam.axisY);
                Camera.main.transform.parent = prevParent;
            }
        }
    }
}