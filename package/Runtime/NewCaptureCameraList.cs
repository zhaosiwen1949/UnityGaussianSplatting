using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace GaussianSplatting.Runtime
{
    public class NewCaptureCameraList : MonoBehaviour
    {
        public GaussianSplatAsset m_Asset;
        
        [Min(0)][Tooltip("Camera Index")]
        public int m_CameraIndex = 0;

        void Update()
        {
            if (Input.GetKeyDown(KeyCode.Space))
            {
                var cam = m_Asset.cameras[m_CameraIndex];
                UpdateCamera(cam);
            }
            
            if (Input.GetKeyDown(KeyCode.RightArrow))
            {
                m_CameraIndex = Math.Min(m_Asset.cameras.Length - 1, m_CameraIndex + 1);
                var cam = m_Asset.cameras[m_CameraIndex];
                UpdateCamera(cam);
            }
            
            if (Input.GetKeyDown(KeyCode.LeftArrow))
            {
                m_CameraIndex = Math.Max(0, m_CameraIndex - 1);
                var cam = m_Asset.cameras[m_CameraIndex];
                UpdateCamera(cam);
            }
        }
        
        public void CaptureShotForCameraList()
        {
            StartCoroutine(CaptureShotForCameraListCoroutine());
        }

        private IEnumerator CaptureShotForCameraListCoroutine()
        {
            string dir = "Screenshots";
            if (!System.IO.File.Exists(dir))
            {
                System.IO.Directory.CreateDirectory(dir);
            }
            
            
            for (var i = 0; i < m_Asset.cameras.Length; i++)
            {
                yield return new WaitForSeconds(0.01f);
                var cam = m_Asset.cameras[i];
                UpdateCamera(cam);
                
                string path = $"{dir}/cam-{i:0000}.png";
                ScreenCapture.CaptureScreenshot(path);
                yield return null;
            }
        }
        
        private void UpdateCamera(GaussianSplatAsset.CameraInfo cam)
        {
            var selfTr = transform;
            selfTr.localScale = new Vector3(-1.0f, 1.0f, 1.0f);
            var camTr = Camera.main.transform;
            var prevParent = camTr.parent;
            Camera.main.transform.parent = selfTr;
            Camera.main.transform.localPosition = cam.pos;
            Camera.main.transform.localRotation = Quaternion.LookRotation(-1 * cam.axisZ, cam.axisY);
            Camera.main.transform.parent = prevParent;
            Camera.main.fieldOfView = cam.fov;
        }
    }
}
