/******************************************************************************
 * GPUSorting
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 4/28/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Assertions;

namespace GPUInt64Sorting.Runtime
{
    public class NewDeviceRadixSort : NewGPUSortBase
    {
        private int m_kernelInit = -1;
        private int m_kernelUpsweep = -1;
        private int m_kernelScan = -1;
        private int m_kernelDownsweep = -1;

        private readonly bool k_keysOnly;

        public NewDeviceRadixSort(
            ComputeShader compute,
            int allocationSize,
            ref GraphicsBuffer tempKeyBuffer,
            ref GraphicsBuffer tempPayloadBuffer,
            ref GraphicsBuffer tempGlobalHistBuffer,
            ref GraphicsBuffer tempPassHistBuffer) :
            base(
                compute,
                allocationSize)
        {
            InitKernels();
            m_cs.EnableKeyword(m_sortPairKeyword);
            k_keysOnly = false;

            tempKeyBuffer?.Dispose();
            tempPayloadBuffer?.Dispose();
            tempGlobalHistBuffer?.Dispose();
            tempPassHistBuffer?.Dispose();

            tempKeyBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, k_maxKeysAllocated, 8) { name="TempKey" };
            tempPayloadBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, k_maxKeysAllocated, 4) { name="TempPayload" };
            tempGlobalHistBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, k_radix * k_radixPasses, 4) { name="TempGloabalHist" };
            tempPassHistBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, k_radix * DivRoundUp(k_maxKeysAllocated, k_partitionSize), 4) { name="TempPassHist" };
        }

        private void InitKernels()
        {
            bool isValid;

            if (m_cs)
            {
                m_kernelInit = m_cs.FindKernel("InitDeviceRadixSort");
                m_kernelUpsweep = m_cs.FindKernel("Upsweep");
                m_kernelScan = m_cs.FindKernel("Scan");
                m_kernelDownsweep = m_cs.FindKernel("Downsweep");
            }

            isValid =   m_kernelInit >= 0 &&
                        m_kernelUpsweep >= 0 &&
                        m_kernelScan >= 0 &&
                        m_kernelDownsweep >= 0;

            if (isValid)
            {
                if (!m_cs.IsSupported(m_kernelInit) ||
                    !m_cs.IsSupported(m_kernelUpsweep) ||
                    !m_cs.IsSupported(m_kernelScan) ||
                    !m_cs.IsSupported(m_kernelDownsweep))
                {
                    isValid = false;
                }
            }

            Assert.IsTrue(isValid);
        }

        private void SetStaticRootParameters(
            CommandBuffer _cmd,
            GraphicsBuffer _numArgsBuffer,
            GraphicsBuffer _passHistBuffer,
            GraphicsBuffer _globalHistBuffer)
        {
            // _cmd.SetComputeIntParam(m_cs, "e_numKeys", numKeys);
            // _cmd.SetComputeIntParam(m_cs, "e_threadBlocks", numThreadBlocks);
            
            _cmd.SetComputeBufferParam(m_cs, m_kernelInit, "b_globalHist", _globalHistBuffer);

            _cmd.SetComputeBufferParam(m_cs, m_kernelUpsweep, "b_passHist", _passHistBuffer);
            _cmd.SetComputeBufferParam(m_cs, m_kernelUpsweep, "b_globalHist", _globalHistBuffer);
            _cmd.SetComputeBufferParam(m_cs, m_kernelUpsweep, "e_numArgs", _numArgsBuffer);

            _cmd.SetComputeBufferParam(m_cs, m_kernelScan, "b_passHist", _passHistBuffer);
            _cmd.SetComputeBufferParam(m_cs, m_kernelScan, "e_numArgs", _numArgsBuffer);

            _cmd.SetComputeBufferParam(m_cs, m_kernelDownsweep, "b_passHist", _passHistBuffer);
            _cmd.SetComputeBufferParam(m_cs, m_kernelDownsweep, "b_globalHist", _globalHistBuffer);
            _cmd.SetComputeBufferParam(m_cs, m_kernelDownsweep, "e_numArgs", _numArgsBuffer);
        }

        private void Dispatch(
            CommandBuffer _cmd,
            int frameCount,
            int frameIndex,
            GraphicsBuffer _numArgsBuffer,
            GraphicsBuffer _toSort,
            GraphicsBuffer _toSortPayload,
            GraphicsBuffer _alt,
            GraphicsBuffer _altPayload)
        {
            const uint argsOffset = 8;
            _cmd.DispatchCompute(m_cs, m_kernelInit, 2, 1, 1);

            int framePasses = k_radixPasses / frameCount;
            
            for (int i = 0; i < framePasses; i++ )
            {
                int radixPass = frameIndex * framePasses + i;
                int radixShift = radixPass * k_log_radix;
                
                var sortList = radixPass % 2 == 0 ? _toSort : _alt;
                var sortPayloadList = radixPass % 2 == 0 ? _toSortPayload : _altPayload;
                var altList = radixPass % 2 == 0 ? _alt : _toSort;
                var altPayloadList = radixPass % 2 == 0 ? _altPayload : _toSortPayload;
                
                _cmd.SetComputeIntParam(m_cs, "e_radixShift", radixShift);
                _cmd.SetComputeIntParam(m_cs, "e_argsOffset", (int)argsOffset / 4 - 1);

                _cmd.SetComputeBufferParam(m_cs, m_kernelUpsweep, "b_sort", sortList);
                // _cmd.SetComputeBufferParam(m_cs, m_kernelUpsweep, "b_sort", _toSort);
                _cmd.DispatchCompute(m_cs, m_kernelUpsweep, _numArgsBuffer, argsOffset);
                
                _cmd.DispatchCompute(m_cs, m_kernelScan, k_radix, 1, 1);
                
                _cmd.SetComputeBufferParam(m_cs, m_kernelDownsweep, "b_sort", sortList);
                _cmd.SetComputeBufferParam(m_cs, m_kernelDownsweep, "b_sortPayload", sortPayloadList);
                _cmd.SetComputeBufferParam(m_cs, m_kernelDownsweep, "b_alt", altList);
                _cmd.SetComputeBufferParam(m_cs, m_kernelDownsweep, "b_altPayload", altPayloadList);
                _cmd.DispatchCompute(m_cs, m_kernelDownsweep, _numArgsBuffer, argsOffset);
                
                // _cmd.SetComputeBufferParam(m_cs, m_kernelDownsweep, "b_sort", _toSort);
                // _cmd.SetComputeBufferParam(m_cs, m_kernelDownsweep, "b_sortPayload", _toSortPayload);
                // _cmd.SetComputeBufferParam(m_cs, m_kernelDownsweep, "b_alt", _alt);
                // _cmd.SetComputeBufferParam(m_cs, m_kernelDownsweep, "b_altPayload", _altPayload);
                // _cmd.DispatchCompute(m_cs, m_kernelDownsweep, _numArgsBuffer, argsOffset);
                //
                // (_toSort, _alt) = (_alt, _toSort);
                // (_toSortPayload, _altPayload) = (_altPayload, _toSortPayload);
            }
        }
        
        private void AssertChecksPairs(System.Type _keyType, System.Type _payloadType)
        {
            Assert.IsFalse(k_keysOnly);
            Assert.IsTrue(
                _keyType == typeof(uint)    ||
                _keyType == typeof(float)   ||
                _keyType == typeof(int)     ||
                _keyType == typeof(ulong));
            Assert.IsTrue(
                _payloadType == typeof(uint)    || 
                _payloadType == typeof(float)   || 
                _payloadType == typeof(int));
        }

        public void Sort(
            CommandBuffer cmd,
            int frameCount,
            int frameIndex,
            GraphicsBuffer sortSize,
            GraphicsBuffer toSort,
            GraphicsBuffer toSortPayload,
            GraphicsBuffer tempKeyBuffer,
            GraphicsBuffer tempPayloadBuffer,
            GraphicsBuffer tempGlobalHistBuffer,
            GraphicsBuffer tempPassHistBuffer,
            System.Type keyType,
            System.Type payloadType,
            bool shouldAscend)
        {
            AssertChecksPairs(keyType, payloadType);
            SetKeyTypeKeywords(cmd, keyType);
            SetPayloadTypeKeywords(cmd, payloadType);
            SetAscendingKeyWords(cmd, shouldAscend);
            SetStaticRootParameters(
                cmd,
                sortSize,
                tempPassHistBuffer,
                tempGlobalHistBuffer);
            Dispatch(cmd, frameCount, frameIndex, sortSize, toSort, toSortPayload, tempKeyBuffer, tempPayloadBuffer);
        }
    }
}