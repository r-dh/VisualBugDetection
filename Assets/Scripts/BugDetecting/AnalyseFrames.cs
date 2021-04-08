using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using UnityEngine;
using System.Collections;
using UnityEngine.Experimental.Rendering;
using System.Diagnostics;
using Unity.Jobs;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine.Rendering;
using System.Buffers;

namespace Assets.Scripts.BugDetecting
{
    class AnalyseFrames : MonoBehaviour
    {
        public bool Record = false;
        public float Interval = 0.5f;
        public List<OnnxModel> models;

        private Rect rect;
        private RenderTexture renderTexture;
        private Texture2D screenShot;
        private readonly int WIDTH = 224;
        private readonly int HEIGHT = 224;
        
        private float dt = 0f;

        List<OnnxBugDetector> bugDetectors = new List<OnnxBugDetector>();

        public void Start()
        {
            if (models.Count == 0)
            {
                models.Add(new OnnxModel("Assets/AIModels/EfficientNetB3_viking_converted.onnx", "MissingTex", "input_3:0"));
            }

            bugDetectors.AddRange(models.Select(model => new OnnxBugDetector(model)));

            rect = new Rect(0, 0, WIDTH, HEIGHT);
            renderTexture = new RenderTexture(WIDTH, HEIGHT, 24);
            screenShot = new Texture2D(WIDTH, HEIGHT, TextureFormat.RGB24, false);
        }
        IEnumerator RecordFrame()
        {
            yield return new WaitForEndOfFrame();

            Camera camera = this.GetComponent<Camera>();
            camera.targetTexture = renderTexture;
            camera.Render();

            RenderTexture.active = renderTexture;
            screenShot.ReadPixels(rect, 0, 0);

            camera.targetTexture = null;
            RenderTexture.active = null;

            yield return new WaitForEndOfFrame(); //Wait for 1 frame between ReadPixels() and GetPixels32() to prevent choking GPU

            _ = EvaluateResult(screenShot); //fire-and-forget
        }

        public async Task EvaluateResult(Texture2D frame)
        {
            Color32[] pixelData32 = frame.GetPixels32();
            byte[] bytes = frame.EncodeToJPG(); //get jpg before inference, otherwise byebye frame

            Tensor<float> data = await Task.Run(() => bugDetectors[0].Color32ToTensor(pixelData32));

            var result = await Task.WhenAll(bugDetectors.Select(bugdetector => bugdetector.Infer(data)));
#if UNITY_EDITOR
            UnityEngine.Debug.Log($"bugs found: { result.Count(r => r == true)}");
#endif
            if (result.Contains(true))
            {
                string positives = "";
                for (int i = 0; i < result.Length; i++)
                {
                    if (result[i])
                    {
                        positives += bugDetectors[i].ModelName + "_";
                    }
                }

                System.DateTime now = System.DateTime.Now;
                string time_stamp = string.Format("{0}{1}{2}{3}{4}{5}",
                    now.Year,
                    now.Month,
                    now.Day,
                    now.Hour,
                    now.Minute,
                    now.Second);
                string path = string.Format($"{Application.persistentDataPath}/colorbugs/{positives}_{time_stamp}.jpg");

                File.WriteAllBytes(path, bytes);
            }
        }

        public void LateUpdate()
        {
            if (Record)
            {
                if (dt > Interval)
                {
                    StartCoroutine(RecordFrame());
                    dt = 0;
                }
                dt += Time.deltaTime;
            }
        }
    }
}
